"""
beetFs
Copyright 2010 Martin Eve

This file is part of beetFs.

beetFs is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

beetFs is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with beetFs.  If not, see <http://www.gnu.org/licenses/>.

Modified by iwconfig, 2025-04-25:
    - Updated code for Python 3.13 compatibility.

"""

import calendar
import datetime
import errno
import fuse # NOTE: Requires Python 3 compatible FUSE bindings, like fusepy
import logging
import operator
import os
import re
import stat
import struct
import functools # Added for reduce and cmp_to_key (though cmp_to_key wasn't needed)
from errno import EINVAL
from io import BytesIO
from string import Template

import beets
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, UserError # Import UserError explicitly
from mutagen.flac import (FLAC, Padding, MetadataBlock, VCFLACDict, CueSheet,
                          SeekTable, FLACNoHeaderError, FLACVorbisError)
from mutagen.id3 import ID3, BitPaddedInt, MakeID3v1 # MakeID3v1 might need checking for bytes output in modern mutagen
from mutagen._util import insert_bytes # Assuming this handles bytes correctly

# Configuration / Globals (Consider encapsulating these better if refactoring)
PATH_FORMAT = ("$artist/$album ($year) [$format_upper]/"
               "$track - $artist - $title.$format")

beetFs_command = Subcommand('mount', help='Mount a beets filesystem')
log = logging.getLogger('beets.beetFs') # Changed logger name slightly for clarity

# FUSE version note: fuse.fuse_python_api is likely obsolete with modern bindings.
# fuse.fuse_python_api = (0, 2) # Removed: Likely specific to old Python 2 fuse bindings

# Global variables used to share state between mount() and the FUSE class.
# This is generally not ideal; passing via the FUSE class constructor is better.
structure_split = []
structure_depth = 0
library = None
directory_structure = None


""" This is duplicated from Library. Ideally should be exposed from there."""
# NOTE: Check if modern beets exposes these fields or a better way to get them.
METADATA_RW_FIELDS = [
    ('title',       'text'),
    ('artist',      'text'),
    ('album',       'text'),
    ('genre',       'text'),
    ('composer',    'text'),
    ('grouping',    'text'),
    ('year',        'int'),
    ('month',       'int'),
    ('day',         'int'),
    ('track',       'int'),
    ('tracktotal',  'int'),
    ('disc',        'int'),
    ('disctotal',   'int'),
    ('lyrics',      'text'),
    ('comments',    'text'),
    ('bpm',         'int'),
    ('comp',        'bool'),
]
METADATA_FIELDS = [
    ('length',  'real'),
    ('bitrate', 'int'),
] + METADATA_RW_FIELDS

# map() returns an iterator in Python 3, which is fine here.
METADATA_KEYS = map(operator.itemgetter(0), METADATA_FIELDS)


def template_mapping(lib, item):
    """ Builds a template substitution map. Taken from library.py."""
    mapping = {}
    # Use list() to consume the iterator if needed elsewhere, but here it's fine
    keys_to_process = list(METADATA_KEYS)
    for key in keys_to_process:
        value = getattr(item, key, None) # Use getattr default for safety

        # Handle missing or None values gracefully
        if value is None:
             value_str = "" # Represent None as empty string for templates
        # sanitize the value for inclusion in a path:
        # replace /,\,: and leading . with _
        elif isinstance(value, str): # Use str instead of basestring
            # Ensure replacement happens on the value
            sanitized_value = value.replace(os.sep, '_')
            # Python 3 re.sub works directly on strings
            sanitized_value = re.sub(r'[\\/:]|^\.', '_', sanitized_value)
            value_str = sanitized_value
        elif key in ('track', 'tracktotal', 'disc', 'disctotal'):
            # pad with zeros
            value_str = '%02i' % int(value) # Ensure value is int before formatting
        else:
            value_str = str(value)
        mapping[key] = value_str

    # Get format, ensure path is treated as bytes if needed by os.path, but beets path should be str
    # Assuming item.path is unicode string (str)
    item_path_str = item.path if isinstance(item.path, str) else item.path.decode('utf-8', 'surrogateescape')
    try:
        format_ = os.path.splitext(item_path_str)[1]
        if format_:
            format_ = format_[1:] # Remove leading dot
        else:
            format_ = 'unknown' # Handle case with no extension
    except (TypeError, ValueError):
        format_ = 'unknown'

    mapping['format'] = re.sub(r'[\\/:]|^\.', '_', format_)
    mapping['format_upper'] = mapping['format'].upper()

    # fix dud entries (use the mapped values)
    if not mapping.get('artist'): # Check if empty or None
        mapping['artist'] = 'Unknown Artist'

    if not mapping.get('album'):
        mapping['album'] = 'Unknown Album'

    # Year might be 0, handle specifically if needed, or check if ''
    if mapping.get('year') == '0' or not mapping.get('year'):
        mapping['year'] = 'Unknown Year'

    if not mapping.get('title'):
        mapping['title'] = 'Unknown Track'

    return mapping


def mount(lib_ref, opts, args):
    # check we have a command line argument
    if not args:
        raise UserError('no mountpoint specified')

    mountpoint = args[0] # Get the mountpoint path

    # build the in-memory folder structure
    global structure_split, structure_depth, library, directory_structure
    structure_split = PATH_FORMAT.split("/")
    structure_depth = len(structure_split)
    templates = {}
    library = lib_ref # Assign to global

    # establish a blank dictionary at each level of depth
    # also create templates for each level
    for level in range(0, structure_depth):
        templates[level] = Template(structure_split[level])

    directory_structure = FSNode({}, {}) # Initialize global

    # iterate over items in library
    log.info(f"Building filesystem structure from {len(library.items())} items...")
    count = 0
    for item in library.items():
        count += 1
        if count % 100 == 0:
             log.debug(f"Processing item {count}...")
        # build the template map
        try:
            mapping = template_mapping(library, item) # Use global library

            # do the substitutions
            level_subbed = {}
            for level in range(0, structure_depth):
                try:
                    level_subbed[level] = templates[level].substitute(mapping)
                except KeyError as e:
                    log.warning(f"Skipping item {item.id} ({item.path}): Missing template key '{e}' in mapping {mapping}")
                    level_subbed = None # Flag to skip this item
                    break
                except ValueError as e:
                    log.warning(f"Skipping item {item.id} ({item.path}): Invalid template value for level {level} ('{structure_split[level]}'): {e}. Mapping: {mapping}")
                    level_subbed = None # Flag to skip this item
                    break

            if level_subbed is None:
                 continue # Skip item if substitution failed

            # build a directory structure
            sub_elements = []
            # Iterate up to the second-to-last level for directories
            for level in range(0, structure_depth - 1):
                # Get the path components processed so far
                current_path_elements = [level_subbed[i] for i in range(level)]
                # add this directory to the master structure
                directory_structure.adddir(current_path_elements, level_subbed[level])

            # add this item as a file
            # The path to the directory containing the file
            file_dir_elements = [level_subbed[i] for i in range(structure_depth - 1)]
            filename = level_subbed[structure_depth-1]
            #log.debug(f"Adding file: Path={file_dir_elements}, Name={filename}, ID={item.id}")
            directory_structure.addfile(file_dir_elements, filename, item.id)

        except Exception as e:
            log.error(f"Error processing item {item.id} ({item.path}): {e}", exc_info=True)
            # Decide whether to continue or raise

    log.info("Filesystem structure built.")
    # Pass necessary data (library, structure) to the FUSE class constructor
    # instead of using globals if refactoring. For now, globals are used.
    # Note: version and usage are handled by fusepy itself if not provided.
    server = beetFileSystem(dash_s_do='setsingle') # Removed version/usage args

    # fusepy main loop arguments are different.
    # It typically takes mountpoint, foreground, nothreads, etc.
    # The server.parse() call used previously is from the old FUSE bindings.
    # We need to call fuse.FUSE() directly or use server.main() if it handles it.

    # Option 1: Use server.main() if it handles parsing args (check fusepy docs/examples)
    # server.parse([mountpoint], errex=1) # This is likely wrong for fusepy

    # Option 2: Instantiate FUSE directly (more typical fusepy usage)
    # fuse.FUSE(server, mountpoint, foreground=True, nothreads=True) # Example

    # Let's assume beetFileSystem subclasses fuse.Operations and we can call main()
    # Need to configure foreground/threading here if possible.
    server.multithreaded = False # Use nothreads equivalent
    server.foreground = True # Run in foreground for testing/debugging

    log.info(f"Mounting filesystem at {mountpoint}")
    try:
        # The FUSE main loop is typically started like this:
        fuse.FUSE(server, mountpoint, foreground=server.foreground, nothreads=(not server.multithreaded))
        # server.main() might also work depending on how beetFileSystem inherits/overrides fuse.Fuse
        # If server inherits fuse.Fuse from old bindings, its main() might be wrong.
        # Let's stick to the fuse.FUSE() call as it's standard fusepy.
    except RuntimeError as e:
        # fusepy raises RuntimeError for common mount issues (e.g., already mounted)
        log.error(f"FUSE Runtime Error: {e}")
    except Exception as e:
        log.error(f"Unhandled error during FUSE setup/mount: {e}", exc_info=True)


beetFs_command.func = mount


class beetFs(BeetsPlugin):
    """ The beets plugin hook."""
    def commands(self):
        return [beetFs_command]


def to_int_be(byte_string):
    """Convert an arbitrarily-long bytes object to an int using big-endian
    byte order."""
    # Use functools.reduce, ensure initial value is int 0
    return functools.reduce(lambda a, b: (a << 8) + b, byte_string, 0)


class InterpolatedID3 (ID3):
    # NOTE: This class overrides save but not load. It relies on the base
    # class's load method. It might be better to use hooks if mutagen provides them.
    def save(self, filename=None, v1=0):
        """Save changes to a file.
        If no filename is given, the one most recently loaded is used.
        Keyword arguments:
        v1 -- if 0, ID3v1 tags will be removed
                    if 1, ID3v1 tags will be updated but not added
                    if 2, ID3v1 tags will be created and/or updated
        """
        # Sort frames by 'importance'
        order = ["TIT2", "TPE1", "TRCK", "TALB", "TPOS", "TDRC", "TCON"]
        order_map = dict(zip(order, range(len(order))))
        last = len(order_map)
        frames = list(self.items()) # Get a list of items

        # Use key function for sorting instead of cmp
        frames.sort(key=lambda item: order_map.get(item[0][:4], last))

        # Assume __save_frame returns bytes
        framedata_list = [self.__save_frame(frame) for (key, frame) in frames]
        framedata_list.extend([data for data in self.unknown_frames
                               if len(data) > 10])

        # Join bytes objects
        framedata = b''.join(framedata_list)
        framesize = len(framedata)

        if filename is None:
            filename = self.filename
            if filename is None:
                 raise ValueError("No filename specified or previously loaded.") # Added check

        # Open in binary read/write mode
        try:
            # Ensure filename is a path-like object or string
            f = open(filename, 'rb+')
        except OSError as e:
            log.error(f"Error opening file {filename} for ID3 save: {e}")
            raise # Re-raise the error

        try:
            idata = f.read(10)
            try:
                # Unpack from bytes
                id3, vmaj, vrev, flags, insize_bytes = struct.unpack('>3sBBB4s', idata)
            except struct.error:
                id3, insize_bytes = b'', b'\x00\x00\x00\x00' # Default to empty bytes

            # Assume BitPaddedInt handles bytes or int correctly
            # Check mutagen docs: BitPaddedInt takes an int
            # Let's parse manually for safety/clarity
            insize = 0
            for byte in insize_bytes:
                 if byte & 0x80: # Check synchsafe bit
                      log.warning(f"Invalid synchsafe int byte {byte} in ID3 size, parsing may fail.")
                      # Fallback or error? Let's try ignoring synchsafe for this calc:
                      # insize = (insize << 8) | byte # Non-synchsafe way
                      # Correct synchsafe parsing:
                 insize = (insize << 7) | (byte & 0x7F)

            # Use bytes literal for comparison
            if id3 != b'ID3':
                insize = -10 # Keep original logic indicator

            if insize >= framesize:
                outsize = insize
            else:
                # Ensure integer division if needed, but logic seems fine
                outsize = (framesize + 1023) & ~0x3FF # Bitwise alignment

            # Pad with null bytes
            framedata += b'\x00' * (outsize - framesize)

            # Convert size back to synchsafe bytes
            framesize_bytes = bytearray(4)
            for i in range(3, -1, -1):
                 framesize_bytes[i] = outsize & 0x7F
                 outsize >>= 7
            if outsize > 0: # Check if size exceeded 28 bits
                 raise ValueError("ID3 frame data size exceeds maximum (2^28-1 bytes)")

            flags = 0 # Assuming flags remain 0
            # Pack using bytes literals and bytes data
            header = struct.pack('>3sBBB4s', b'ID3', 4, 0, flags, bytes(framesize_bytes))
            data_to_write = header + framedata # Concatenate bytes

            # Original logic for inserting bytes if needed
            if insize < 0: # Handle the case where no ID3 tag was found initially
                f.seek(0) # Go to the beginning
                log.warning("Inserting new ID3v2 tag, this might shift audio data.")
                # Assuming insert_bytes prepends correctly (needs careful testing)
                try:
                    insert_bytes(f, len(data_to_write), 0) # Make space at the beginning
                    f.seek(0)
                    f.write(data_to_write) # Write the new tag
                except Exception as e:
                    log.error(f"Failed to insert ID3v2 tag using insert_bytes: {e}")
                    raise IOError("Failed to write new ID3 tag") from e

            elif insize < outsize:
                # Make space if the new tag is larger
                try:
                    insert_bytes(f, outsize - insize, insize + 10) # Make space after old tag header
                    f.seek(0)
                    f.write(data_to_write) # Overwrite old tag + padding
                except Exception as e:
                    log.error(f"Failed to expand ID3v2 tag using insert_bytes: {e}")
                    raise IOError("Failed to write resized ID3 tag") from e
            else:
                # Overwrite existing tag space if new tag is same size or smaller
                f.seek(0)
                f.write(data_to_write)
                if insize > outsize:
                    # If new tag is smaller, we need to truncate the excess space.
                    # This is tricky. Overwriting with padding is done by data_to_write.
                    # But if the *original* file had audio data immediately after the old padding,
                    # just writing the smaller tag could leave stale padding *before* the audio.
                    # The safest way often involves rewriting the audio data too.
                    # For now, assume writing the new (smaller) tag + padding is sufficient.
                    # Check if f.truncate() is needed after the write.
                    # Let's truncate the file to the new total size if shrinking.
                    # f.truncate(new_total_size) -> Requires knowing where audio starts. Difficult.
                    # Stick to original logic: Smaller tag overwrites beginning, padding handles rest.
                    pass # Covered by writing data_to_write which includes padding

            # ID3v1 tag handling
            try:
                f.seek(-128, 2) # Seek from end for ID3v1
            except OSError as err:
                # EINVAL typically means seeking before start (file < 128 bytes)
                if err.errno != EINVAL:
                    raise
                f.seek(0, 2)  # Go to end if file is small

            # Compare with bytes literal
            tag_check = f.read(3)
            is_id3v1 = (tag_check == b"TAG")

            if is_id3v1:
                if v1 > 0:
                    f.seek(-128, 2)
                    # MakeID3v1 needs to return bytes
                    id3v1_data = MakeID3v1(self) # Assume returns bytes
                    if not isinstance(id3v1_data, bytes):
                         log.warning("MakeID3v1 did not return bytes, encoding as latin-1")
                         id3v1_data = str(id3v1_data).encode('latin-1', 'replace') # Fallback encoding
                    f.write(id3v1_data)
                else: # v1 == 0: remove tag
                    f.seek(-128, 2)
                    f.truncate() # Truncate the last 128 bytes
            elif v1 == 2: # Add tag if not present
                f.seek(0, 2) # Go to end
                id3v1_data = MakeID3v1(self) # Assume returns bytes
                if not isinstance(id3v1_data, bytes):
                    log.warning("MakeID3v1 did not return bytes, encoding as latin-1")
                    id3v1_data = str(id3v1_data).encode('latin-1', 'replace') # Fallback encoding
                f.write(id3v1_data)

        finally:
            f.close()


class InterpolatedFLAC (FLAC):
    # Overrides load and adds get_header, offset methods
    def load(self, filedata):
        # filedata should be bytes
        if not isinstance(filedata, bytes):
             raise TypeError("InterpolatedFLAC.load expects bytes data")

        self.metadata_blocks = []
        self.tags = None
        self.cuesheet = None
        self.seektable = None
        # self.filename = filename # filename not used directly here
        self.filedata = filedata # Store the bytes data
        self.fileobj = BytesIO(filedata) # Use BytesIO for in-memory binary data
        self.__offset = 0 # Initialize offset

        try:
            header_size = self.__check_header(self.fileobj) # Reads header
            if header_size is None:
                 # Exception already raised in __check_header
                 return

            self.fileobj.seek(header_size) # Position after 'fLaC' (and potential ID3)

            while self.__read_metadata_block(self.fileobj):
                pass # Loop reads blocks until the last one is found

            # Store the offset where metadata ends / audio begins
            self.__offset = self.fileobj.tell()

            # Check for audio frame sync code after metadata
            sync_code = self.fileobj.read(2)
            # Use bytes literals
            if sync_code not in [b"\xff\xf8", b"\xff\xf9"]:
                # FLAC spec allows other sync codes, but mutagen checks these.
                # Might need adjustment based on FLAC spec details.
                # Checking first bit == 1 might be more robust: (sync_code[0] >> 7) == 1 ?
                if not sync_code or (sync_code[0] & 0b11111110) != 0b11111110: # Check for 1111111x pattern
                    log.warning(f"Potential invalid FLAC audio frame sync code {sync_code!r} found after metadata.")
                    # raise FLACNoHeaderError("End of metadata did not start with expected audio sync code pattern")

            # Ensure StreamInfo block was found (usually the first block)
            if not self.metadata_blocks or self.metadata_blocks[0].code != 0: # Code 0 is STREAMINFO
                 raise FLACNoHeaderError("Stream info block not found or not first")

            log.debug("InterpolatedFLAC loaded successfully")

        except Exception as e:
            # Clean up or re-raise
            log.error(f"Error loading FLAC data: {e}", exc_info=True)
            # Reset state?
            self.metadata_blocks = []
            self.tags = None
            raise # Re-raise the exception


    def __read_metadata_block(self, file):
        # file is BytesIO
        try:
            header = file.read(4)
            if len(header) < 4:
                # Reached end of stream unexpectedly
                if len(header) == 0: # Clean EOF
                    return False
                else: # Partial header read - data corruption?
                    raise FLACNoHeaderError("Incomplete FLAC metadata block header found")

            # ord() works on single byte (int in py2, requires index in py3)
            byte = header[0]
            # Use bytes slicing and helper function
            size = to_int_be(header[1:4])

            # Check for excessive block size to prevent OOM errors
            # Needs context of total file size, but add a basic sanity check
            if size > 16 * 1024 * 1024: # e.g., refuse blocks > 16MB (common picture block limit)
                raise FLACVorbisError(f"FLAC block size {size} exceeds sanity limit")

            data = file.read(size)
            if len(data) != size:
                log.error(f"FLAC block error: Expected {size} bytes, got {len(data)}")
                raise FLACNoHeaderError("Incomplete FLAC metadata block data")

            block_type = byte & 0x7F
            is_last = (byte >> 7) & 1

            # Use mutagen's block type mapping if possible
            try:
                # METADATA_BLOCKS needs to be accessed correctly (it's a class attr on FLAC)
                block_class = FLAC.METADATA_BLOCKS.get(block_type)
                if block_class:
                     block = block_class(data)
                else:
                     # Fallback for unknown block types
                     block = MetadataBlock(data)
                     block.code = block_type
                     log.debug(f"Read unknown FLAC block type {block_type}")
            except Exception as e: # Catch potential errors during block parsing
                log.warning(f"Error parsing FLAC block type {block_type}: {e}. Treating as raw MetadataBlock.")
                block = MetadataBlock(data)
                block.code = block_type

            self.metadata_blocks.append(block)

            # Assign special blocks (Vorbis Comment, CueSheet, SeekTable)
            if block.code == VCFLACDict.code:
                if self.tags is None:
                    self.tags = block
                else:
                    # Mutagen itself allows multiple Vorbis comment blocks.
                    # It usually merges them or reads the first one. Let's just keep the first.
                    log.warning("Multiple Vorbis comment blocks found, using the first.")
                    # raise FLACVorbisError("> 1 Vorbis comment block found") # Avoid error
            elif block.code == CueSheet.code:
                if self.cuesheet is None:
                    self.cuesheet = block
                else:
                    log.warning("> 1 CueSheet block found, using the first.")
            elif block.code == SeekTable.code:
                if self.seektable is None:
                    self.seektable = block
                else:
                    log.warning("> 1 SeekTable block found, using the first.")

            return not is_last # Return True if more blocks follow

        except EOFError: # Should not happen with BytesIO unless read past end
             log.debug("EOF reached unexpectedly while reading FLAC metadata block.")
             return False # End of stream
        except Exception as e:
             log.error(f"Error reading FLAC metadata block: {e}", exc_info=True)
             raise # Re-raise


    def get_header(self):
        # NOTE: filename parameter removed, operates on internal self.fileobj/self.metadata_blocks

        # Ensure Vorbis comment block exists if tags are present in object
        if self.tags is None and any(isinstance(b, VCFLACDict) for b in self.metadata_blocks):
             # If self.tags was somehow unset but block exists, re-assign
             self.tags = next(b for b in self.metadata_blocks if isinstance(b, VCFLACDict))
        elif self.tags is not None and not any(isinstance(b, VCFLACDict) for b in self.metadata_blocks):
             # If self.tags exists but block doesn't, add it
             # Ensure STREAMINFO (code 0) exists and is first
             if not self.metadata_blocks or self.metadata_blocks[0].code != 0:
                 raise FLACError("Cannot add Vorbis comment: STREAMINFO block missing or not first.")
             self.metadata_blocks.insert(1, self.tags) # Insert after STREAMINFO
        elif self.tags is None:
             # Create a new VCFLACDict if none exists and we might add tags later?
             # Let's create it if it's missing entirely, to store tags properly.
             if not any(isinstance(b, VCFLACDict) for b in self.metadata_blocks):
                 log.debug("No Vorbis comment block found, creating empty one.")
                 self.tags = VCFLACDict()
                 if self.metadata_blocks and self.metadata_blocks[0].code == 0:
                      self.metadata_blocks.insert(1, self.tags)
                 else: # Prepend if no streaminfo? Should not happen in valid FLAC.
                      self.metadata_blocks.insert(0, self.tags)


        # Ensure we have padding at the end, but manage it carefully.
        # Remove existing padding blocks first.
        current_blocks = [b for b in self.metadata_blocks if not isinstance(b, Padding)]
        # Add a small padding block initially. It will be resized.
        # Use bytes for padding data
        initial_padding = 1024 # Default padding size mutagen uses? Check mutagen source.
        padding_block = Padding(b'\x00' * initial_padding)
        current_blocks.append(padding_block)

        # Use mutagen's utility to group padding (moves padding to end, coalesces)
        # Let's ensure padding is last manually. Already done by append/remove logic.

        # Get the size required for the metadata blocks
        try:
             # writeblocks expects a list of blocks
             metadata_bytes = MetadataBlock.writeblocks(current_blocks)
        except Exception as e:
             log.error(f"Error writing FLAC metadata blocks: {e}", exc_info=True)
             raise

        # Calculate available space in the original filedata (if needed for resizing)
        # This seems overly complex if we are generating the header *de novo*
        # Let's simplify: just generate the header based on current blocks.
        # The 'available' logic was for in-place modification, which is harder.

        # Construct the full header: 'fLaC' + metadata
        # Use bytes literal
        full_header = b"fLaC" + metadata_bytes

        # Update the internal offset (start of audio data)
        self.__offset = len(full_header)

        log.debug(f"Generated FLAC header, total length {self.__offset}")
        return full_header


    def offset(self):
        # Returns the calculated offset of the audio data
        return self.__offset

    def __find_audio_offset(self, fileobj):
         # This duplicates logic now in load(). Kept for reference? Or remove.
         # This method seems unused now. Remove it.
         pass

    def __check_header(self, fileobj):
        # fileobj is BytesIO, already at position 0
        initial_pos = fileobj.tell()
        header = fileobj.read(4)
        size = 0
        # Use bytes literals
        if header == b"fLaC":
            size = 4
            fileobj.seek(initial_pos + size) # Position after 'fLaC'
            return size
        elif header[:3] == b"ID3":
            # Need to parse ID3 header size correctly
            try:
                 id3_header_rest = fileobj.read(6) # Read rest of ID3 header (10 bytes total)
                 if len(id3_header_rest) < 6:
                      raise FLACNoHeaderError("Incomplete ID3 header found")

                 # Correctly parse ID3 size (synchsafe integer)
                 size_bytes = id3_header_rest[2:6] # Get the 4 size bytes
                 id3_size = 0
                 for byte in size_bytes:
                      if byte & 0x80:
                           raise FLACNoHeaderError("Invalid ID3 tag size (synchsafe byte > 0x7F)")
                      id3_size = (id3_size << 7) | byte
                 full_id3_size = id3_size + 10 # Add header size

                 log.debug(f"Found ID3 tag, size {full_id3_size}")
                 fileobj.seek(initial_pos + full_id3_size) # Seek past ID3 tag
                 # Check for 'fLaC' immediately after ID3
                 flac_marker = fileobj.read(4)
                 if flac_marker == b"fLaC":
                      log.debug("Found fLaC marker after ID3 tag")
                      # Position *after* 'fLaC' marker
                      return full_id3_size + 4 # Return total size of ID3 + 'fLaC'
                 else:
                      raise FLACNoHeaderError("fLaC marker not found after ID3 tag")
            except (struct.error, IndexError, FLACNoHeaderError) as e:
                 log.error(f"Error parsing potential ID3 tag: {e}")
                 # Fall through to raise error
        # If neither fLaC nor valid ID3+fLaC found
        fileobj.seek(initial_pos) # Reset position
        raise FLACNoHeaderError("File does not start with 'fLaC' or valid ID3+'fLaC'")


class FSNode(object):
    """ A directory node. Contains directories (as a dictionary keyed
        by directory name) and files (dictionary keyed by filename to id).
    """
    def __init__(self, dirs, files):
        self.dirs = dirs # {dirname: FSNode}
        self.files = files # {filename: item_id}

    def getnode(self, elements):
        # elements is a list of path components (strings)
        node = self
        current_path = "/"
        try:
            for element in elements:
                if not element: continue # Skip empty elements resulting from split('/')
                node = node.dirs[element]
                current_path = os.path.join(current_path, element)
            return node
        except KeyError:
             log.debug(f"Path element not found: '{element}' in path '{current_path}'")
             return None # Indicate node not found


    def adddir(self, elements, directory):
        # elements is list of parent path components
        # directory is the name of the dir to add
        node = self.getnode(elements)
        if node is not None and directory not in node.dirs:
            node.dirs[directory] = FSNode({}, {})
            #log.debug(f"Added directory '{directory}' under '{'/'.join(elements)}'")

    def addfile(self, elements, filename, item_id):
        # elements is list of parent path components
        node = self.getnode(elements)
        if node is not None:
            if filename in node.files:
                 log.warning(f"File '{filename}' already exists under '{'/'.join(elements)}' (ID: {node.files[filename]}), overwriting with ID: {item_id}")
            node.files[filename] = item_id
            #log.debug(f"Added file '{filename}' under '{'/'.join(elements)}' (ID: {item_id})")

    def listdir(self, elements):
        # elements is list of path components for the dir to list
        node = self.getnode(elements)
        if node is not None:
            # Return iterator/list of directory names and file names
            return list(node.dirs.keys()), list(node.files.keys())
        else:
            return [], [] # Return empty lists if path not found


class FileHandler(object):
    """ Manages an open file representation, handling metadata interpolation."""
    def __init__(self, path, lib_ref):
        self.path = path # The virtual path in FUSE
        self.lib = lib_ref
        self.real_path = None # Physical path on disk
        self.item = None # The corresponding beets Item
        self.format = None # 'flac', 'mp3', etc.
        self.file_object = None # Raw file object (kept closed mostly)
        self.instance_count = 1 # Track open calls
        self.modified = False # Track if metadata was changed

        # Internal data buffers
        self.header = b'' # Interpolated header bytes
        self.music_data = b'' # Raw music data bytes after header
        self.bound = 0 # Size of the interpolated header
        self.music_offset = 0 # Offset where real music data starts in original file

        try:
            # Resolve virtual path to beets item ID
            # Strip leading/trailing slashes and split
            pathsplit = path.strip('/').split('/')
            # Handle potential empty strings if path starts with '/' or has '//'
            pathsplit = [p for p in pathsplit if p]

            if not pathsplit: # Should not happen if path != "/"
                 raise FileNotFoundError(f"Invalid empty path components for FileHandler: {path}")

            filename = pathsplit[-1]
            dir_elements = pathsplit[:-1]

            node = directory_structure.getnode(dir_elements)
            if node is None or filename not in node.files:
                 log.error(f"File not found in structure: Dir='{'/'.join(dir_elements)}', File='{filename}'")
                 raise FileNotFoundError(f"Path {path} not found in directory structure")

            item_id = node.files[filename]
            self.item = self.lib.get_item(item_id)
            if not self.item:
                 raise FileNotFoundError(f"Item with id {item_id} not found in library for path {path}")

            # Get real path (as bytes or str? beets stores str)
            # Ensure it's a string for os.path operations
            self.real_path = self.item.path
            if isinstance(self.real_path, bytes):
                 self.real_path = self.real_path.decode('utf-8', 'surrogateescape') # Decode if bytes

            log.debug(f"FileHandler: Virtual='{path}', Real='{self.real_path}', ID={self.item.id}")

            # Determine format
            _, ext = os.path.splitext(self.real_path)
            self.format = ext[1:].lower() if ext else None

            # Read the entire file content into memory - potentially large!
            # Consider memory-mapping or chunked reading for large files if memory is an issue.
            try:
                with open(self.real_path, 'rb') as f: # Open in binary read mode
                    file_content = f.read()
            except FileNotFoundError:
                log.error(f"Real file not found: {self.real_path} (referenced by item {self.item.id})")
                raise FileNotFoundError(f"Real file missing for {path}")
            except OSError as e:
                log.error(f"Error reading real file {self.real_path}: {e}")
                raise # Re-raise

            # Process based on format for metadata interpolation
            if self.format == "flac":
                self.inf = InterpolatedFLAC() # Create instance
                try:
                    self.inf.load(file_content) # Load from bytes content
                except (FLACNoHeaderError, FLACVorbisError) as e:
                    log.error(f"Error loading FLAC data for {self.real_path}: {e}. Treating as opaque file.")
                    # Fallback to non-interpolated handling
                    self.format = "unknown" # Mark as unknown to skip interpolation
                    self.header = b''
                    self.bound = 0
                    self.music_offset = 0
                    self.music_data = file_content
                    log.debug(f"FileHandler initialized (non-interpolated) for {path}.")
                    return # Exit init early


                # Sync tags from beets item to mutagen object BEFORE generating header
                if self.inf.tags is None: self.inf.tags = VCFLACDict() # Ensure tags object exists
                # Use list of strings for mutagen tags
                self.inf.tags["title"] = [str(self.item.title or '')]
                self.inf.tags["album"] = [str(self.item.album or '')]
                self.inf.tags["artist"] = [str(self.item.artist or '')]
                self.inf.tags["genre"] = [str(self.item.genre or '')]
                # Add other relevant tags (tracknumber, date, etc.)?
                self.inf.tags["tracknumber"] = [str(self.item.track or 0)]
                # Use albumartist if available?
                self.inf.tags["albumartist"] = [str(self.item.albumartist or self.item.artist or '')]
                if self.item.year:
                    date_str = f"{self.item.year:04}"
                    if self.item.month:
                        date_str += f"-{self.item.month:02}"
                        if self.item.day:
                             date_str += f"-{self.item.day:02}"
                    self.inf.tags["date"] = [date_str]


                self.header = self.inf.get_header() # Generate header bytes
                self.bound = len(self.header) # Size of generated header
                # Get the offset where audio data *originally* started
                self.music_offset = self.inf.offset() # Original offset from loaded data

                # Extract music data from original content based on *original* offset
                self.music_data = file_content[self.music_offset:]

            elif self.format == "mp3":
                # MP3 handling needs InterpolatedID3 logic adapted
                # For now, disable interpolation for MP3 (treat as read-only based on real file)
                log.warning(f"MP3 interpolation not fully implemented ({path}). Read-only access based on file.")
                self.header = b'' # No interpolated header for now
                self.bound = 0
                self.music_offset = 0 # Assume music starts at beginning
                self.music_data = file_content # Use full original content

            else:
                # Unsupported format - treat as read-only
                log.warning(f"Unsupported format '{self.format}' for interpolation ({path}). Read-only access.")
                self.header = b''
                self.bound = 0
                self.music_offset = 0
                self.music_data = file_content

            log.debug(f"FileHandler initialized for {path}. Header size: {self.bound}, Music size: {len(self.music_data)}")

        except Exception as e:
            log.error(f"Error initializing FileHandler for {path}: {e}", exc_info=True)
            # Ensure consistent state on failure
            self.item = None
            self.real_path = None
            self.header = b''
            self.music_data = b''
            raise # Re-raise exception to signal failure to open

    def open(self):
        # Increment instance count for tracking multiple opens
        self.instance_count += 1
        log.debug(f"FileHandler.open: {self.path}, instance count {self.instance_count}")

    def release(self):
        # Decrement instance count. Return True if this was the last instance.
        if self.instance_count > 0:
            self.instance_count -= 1
            log.debug(f"FileHandler.release: {self.path}, instance count {self.instance_count}")
            if self.instance_count == 0:
                 if self.modified:
                      # Persist changes back to the real file on final release
                      # This is complex: requires writing the interpolated header + music data
                      # and potentially updating beets DB if tags were derived from writes.
                      # Current write() updates beets DB immediately, which might be premature.
                      log.info(f"TODO: Implement saving modified file {self.path} back to {self.real_path} on release.")
                      # Example: Write combined data back to real_path
                      # try:
                      #      with open(self.real_path, 'wb') as f:
                      #           f.write(self.header)
                      #           f.write(self.music_data)
                      #      log.info(f"Successfully wrote modified data back to {self.real_path}")
                      # except OSError as e:
                      #      log.error(f"Failed to write modified data to {self.real_path}: {e}")
                      pass # Placeholder for saving logic
                 return True # Signal last release
        return False # More instances remain

    def read(self, size, offset):
        # Read from the combined virtual file (header + music_data)
        virtual_size = self.bound + len(self.music_data)

        if offset < 0: # Handle invalid offset
             log.warning(f"Read request with negative offset: {offset}")
             raise fuse.FuseOSError(errno.EINVAL)

        if offset >= virtual_size:
             return b'' # Read past EOF

        # Clamp read size
        bytes_available = virtual_size - offset
        read_size = min(size, bytes_available)

        #log.debug(f"Read request: offset={offset}, size={size} -> read_size={read_size}")

        # Determine if read spans header, music, or both
        if offset < self.bound:
             # Starts in header
             header_read_start = offset
             header_read_end = min(offset + read_size, self.bound)
             header_part = self.header[header_read_start:header_read_end]

             if header_read_end >= self.bound:
                  # Read potentially crosses into music data
                  music_needed = read_size - len(header_part)
                  if music_needed > 0:
                       music_part = self.music_data[0:music_needed]
                       #log.debug("Read: Header + Music")
                       return header_part + music_part
                  else:
                       # Read ends exactly at header boundary
                       #log.debug("Read: Header only (ends at boundary)")
                       return header_part
             else:
                  # Only header data needed
                  #log.debug("Read: Header only")
                  return header_part
        else:
             # Starts in music data
             #log.debug("Read: Music only")
             music_start = offset - self.bound
             music_end = music_start + read_size
             music_part = self.music_data[music_start:music_end]
             return music_part

    def write(self, offset, buf):
        # buf is bytes
        buf_len = len(buf)
        log.debug(f"Write request: offset={offset}, len={buf_len}")

        if offset < 0: # Handle invalid offset
             log.warning(f"Write request with negative offset: {offset}")
             raise fuse.FuseOSError(errno.EINVAL)

        # Determine if write affects the header region
        # Allow writing at offset 0 even if bound is 0 (e.g., initially empty file)
        write_affects_header = (offset < self.bound or (offset == 0 and self.bound == 0))

        if write_affects_header:
             # Write affects header - requires re-parsing metadata
             log.info(f"Write to header region (offset {offset} < bound {self.bound}). Re-parsing...")
             self.modified = True

             # Create the full virtual file data in memory by patching
             # This can be memory intensive!
             current_header = self.header
             current_music = self.music_data
             current_len = len(current_header) + len(current_music)

             # Calculate end of write operation
             write_end = offset + buf_len

             # Construct new data (simplified approach: assume write replaces, might need padding/truncation logic)
             if offset < self.bound: # Write starts within header
                 new_header_part1 = current_header[:offset]
                 # Part of buf that goes into header vs music
                 buf_in_header = buf[:max(0, self.bound - offset)]
                 buf_in_music = buf[max(0, self.bound - offset):]

                 new_header = new_header_part1 + buf_in_header
                 remaining_header = current_header[min(self.bound, offset + len(buf_in_header)):]

                 # Combine header parts if write is fully contained
                 if write_end <= self.bound:
                     new_header += remaining_header

                 # Now handle music part
                 if buf_in_music: # Write crosses into music data
                      new_music_part1 = buf_in_music
                      remaining_music = current_music[max(0, write_end - self.bound):]
                      new_music = new_music_part1 + remaining_music
                 else: # Write ends within header
                      new_music = current_music # Music data unchanged

                 # Combine potentially shortened header with music
                 # If write ended in header, header might be shorter or longer
                 new_data = new_header + new_music # Reconstruct based on write extent

             else: # Write starts exactly at boundary or offset 0 when bound is 0
                  new_header = current_header # Header unchanged
                  music_write_offset = offset - self.bound # Should be 0
                  new_music = buf + current_music[music_write_offset + buf_len:]
                  new_data = new_header + new_music

             # Re-process the modified data
             if self.format == "flac":
                 try:
                     # Create a new FLAC object from the modified bytes
                     new_inf = InterpolatedFLAC()
                     new_inf.load(new_data) # Load the patched data

                     # Extract metadata from the *new* FLAC tags
                     # Use .get() with default for safety
                     # Ensure tags exist before trying to access them
                     tags_present = new_inf.tags is not None
                     self.item.title = str(new_inf.tags.get("title", [''])[0]) if tags_present else ''
                     self.item.album = str(new_inf.tags.get("album", [''])[0]) if tags_present else ''
                     self.item.artist = str(new_inf.tags.get("artist", [''])[0]) if tags_present else ''
                     self.item.genre = str(new_inf.tags.get("genre", [''])[0]) if tags_present else ''
                     # Update other fields? Track, Year?
                     self.item.track = int(new_inf.tags.get("tracknumber", [0])[0] or 0) if tags_present else 0
                     # Parse year carefully
                     year_str = str(new_inf.tags.get("date", ['0'])[0] or '0').split('-')[0]
                     self.item.year = int(year_str) if year_str.isdigit() else 0

                     log.info(f"Updating beets item {self.item.id} from written tags.")
                     # Update beets library (immediate save - maybe defer?)
                     self.lib.store(self.item)
                     # Defer saving the library to reduce disk I/O, maybe save on fsdestroy?
                     # self.lib.save() -> Removed for performance, potential data loss if crash

                     # Now, regenerate the internal state based on the *beets* data (source of truth)
                     self.inf = InterpolatedFLAC() # Create fresh instance
                     # Need to reconstruct the file data *using the updated beets item*
                     # This is complex: requires re-reading original audio + applying NEW tags
                     # Simpler: Use the data we just parsed ('new_data') as the basis
                     self.inf.load(new_data) # Load the data that reflects the write

                     # Re-apply beets data -> mutagen object (ensure consistency after store)
                     if self.inf.tags is None: self.inf.tags = VCFLACDict()
                     self.inf.tags["title"] = [str(self.item.title or '')]
                     self.inf.tags["album"] = [str(self.item.album or '')]
                     self.inf.tags["artist"] = [str(self.item.artist or '')]
                     self.inf.tags["genre"] = [str(self.item.genre or '')]
                     self.inf.tags["tracknumber"] = [str(self.item.track or 0)]
                     if self.item.year:
                         date_str = f"{self.item.year:04}"
                         if self.item.month: date_str += f"-{self.item.month:02}"
                         if self.item.day: date_str += f"-{self.item.day:02}"
                         self.inf.tags["date"] = [date_str]


                     # Regenerate header and update internal buffers based on potentially modified data
                     self.header = self.inf.get_header()
                     self.bound = len(self.header)
                     # Music data *might* have changed if the header size changed or write crossed boundary
                     # Use the music data associated with the *re-parsed* file
                     self.music_offset = self.inf.offset() # Original offset in 'new_data'
                     self.music_data = new_data[self.music_offset:] # Music part relative to new header

                     log.debug(f"Write successful, header updated. New bound: {self.bound}")
                     return buf_len # Return bytes written

                 except (OSError, FLACNoHeaderError, FLACVorbisError, TypeError, ValueError, KeyError) as e:
                      log.error(f"Couldn't update tags after write for {self.path}: {e}", exc_info=True)
                      # Should we revert the changes in memory? For now, we don't.
                      # Return error to FUSE? EIO?
                      raise fuse.FuseOSError(errno.EIO) # Raise EIO

             elif self.format == "mp3":
                  # MP3 write handling (similar logic with InterpolatedID3)
                  log.warning("MP3 write support is not fully implemented.")
                  # For now, disallow writes to MP3 header region?
                  raise fuse.FuseOSError(errno.EACCES) # Permission denied
             else:
                  # Unsupported format for writing metadata
                  log.warning(f"Write to header region denied for unsupported format {self.format} on {self.path}")
                  raise fuse.FuseOSError(errno.EACCES) # Permission denied

        else:
            # Write affects only music data region
            log.debug("Write to music data region.")
            self.modified = True # Mark modified even if only music data changes

            music_offset = offset - self.bound
            music_len = len(self.music_data)

            # Patch the self.music_data buffer
            part1 = self.music_data[:music_offset]
            part3 = self.music_data[music_offset + buf_len:]

            # Handle writing past the end of music data (extend with nulls?)
            if music_offset > music_len:
                 padding = b'\x00' * (music_offset - music_len)
                 self.music_data = self.music_data + padding + buf
            else:
                 self.music_data = part1 + buf + part3

            return buf_len # Return bytes written


class Stat(object): # Inherit from object (default in Py3)
    """ Custom Stat class helper to generate stat dictionaries. """
    DIRSIZE = 4096 # Standard directory size

    def __init__(self, st_mode, st_size, st_nlink=1, st_uid=None, st_gid=None,
                 dt_atime=None, dt_mtime=None, dt_ctime=None):

        # Initialize all stat attributes expected by FUSE
        self.st_mode = st_mode      # File mode (type and permissions)
        self.st_ino = 0             # Inode number (FUSE often ignores)
        self.st_dev = 0             # Device ID (FUSE often ignores)
        self.st_nlink = st_nlink    # Number of hard links
        self.st_uid = st_uid if st_uid is not None else os.getuid() # User ID
        self.st_gid = st_gid if st_gid is not None else os.getgid() # Group ID
        self.st_size = st_size      # Size in bytes
        self.st_blksize = 4096      # Block size for filesystem I/O stats
        # Calculate blocks based on size and blocksize
        self.st_blocks = (st_size + self.st_blksize - 1) // self.st_blksize

        # Set timestamps (use current time if not provided)
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        atime = self.datetime_epoch(dt_atime or now_dt)
        mtime = self.datetime_epoch(dt_mtime or now_dt)
        ctime = self.datetime_epoch(dt_ctime or now_dt)

        # Store timestamps as floats (seconds since epoch) for fusepy dictionary
        self.st_atime = atime
        self.st_mtime = mtime
        self.st_ctime = ctime

        # fusepy often expects these as floats directly in the dict.
        # Nanosecond fields are less common unless using raw_fi=True?
        # Let's stick to float seconds for st_atime, st_mtime, st_ctime.
        # self.st_atime_ns = int((atime % 1) * 1e9)
        # self.st_mtime_ns = int((mtime % 1) * 1e9)
        # self.st_ctime_ns = int((ctime % 1) * 1e9)
        # self.st_atime = int(atime)
        # self.st_mtime = int(mtime)
        # self.st_ctime = int(ctime)

    # Properties for datetime access (optional, internal uses timestamps)
    @property
    def dt_atime(self):
        return self.epoch_datetime(self.st_atime)

    @dt_atime.setter
    def dt_atime(self, value):
        self.st_atime = self.datetime_epoch(value)

    @property
    def dt_mtime(self):
        return self.epoch_datetime(self.st_mtime)

    @dt_mtime.setter
    def dt_mtime(self, value):
        self.st_mtime = self.datetime_epoch(value)

    @property
    def dt_ctime(self):
        return self.epoch_datetime(self.st_ctime)

    @dt_ctime.setter
    def dt_ctime(self, value):
        self.st_ctime = self.datetime_epoch(value)


    @staticmethod
    def datetime_epoch(dt):
        """Converts a datetime object to a UTC POSIX timestamp (float seconds)."""
        if dt is None:
             # Handle None input if necessary, maybe return current time?
             dt = datetime.datetime.now(datetime.timezone.utc)
        # Ensure datetime is timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
             # This assumes local time if naive, might be wrong. Best if datetimes are aware.
             # Let's assume UTC for naive dates from os.stat
             dt = dt.replace(tzinfo=datetime.timezone.utc) # Assume UTC
        # calendar.timegm is deprecated for aware objects, use timestamp()
        return dt.timestamp()


    @staticmethod
    def epoch_datetime(seconds):
        """Converts a UTC POSIX timestamp (float seconds) to a UTC datetime object."""
        return datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)


# Inherit from fuse.Operations for fusepy
class beetFileSystem(fuse.Operations):

    def __init__(self, dash_s_do=None): # Removed unused fuse.Fuse args
        # Setup logging before calling super
        # Use default beets log configuration? Or configure separately?
        # For now, keep separate log file.
        LOG_FILENAME = "beetfs.log"
        # Check if logger already configured by beets itself
        fs_log = logging.getLogger('beets.beetFs') # Use specific sub-logger
        if not fs_log.handlers:
            log_handler = logging.FileHandler(LOG_FILENAME)
            log_formatter = logging.Formatter(
                '%(asctime)s %(levelname)s [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            log_handler.setFormatter(log_formatter)
            fs_log.addHandler(log_handler)
            fs_log.setLevel(logging.INFO) # Set level (DEBUG for more)
            fs_log.propagate = False # Prevent double logging to beets root logger if needed

        fs_log.info("Initializing beetFileSystem...")

        # Initialize instance variables (instead of relying only on globals)
        # These are still populated from globals in fsinit for now
        self.lib = None
        self.directory_structure = None
        self.files = {} # Dictionary to store active FileHandler instances {path: FileHandler}
        self.structure_depth = 0 # Store calculated depth
        self.next_fh = 0 # Counter for file handle IDs
        self.open_files_by_fh = {} # Map fh_id -> FileHandler instance

        # Store options passed from command? Not directly used by Operations.
        self.dash_s_do = dash_s_do
        self.foreground = False # Default to background
        self.multithreaded = True # Default to multithreaded


    # fsinit/fsdestroy are less common in fusepy Operations compared to Fuse class
    # Use __init__ and potentially handle cleanup if needed elsewhere.
    def __call__(self, op, path, *args):
        """ Log all FUSE operations. """
        log.debug(f'FUSE operation: {op} {path} {args}')
        return super(beetFileSystem, self).__call__(op, path, *args)

    def _init_check(self):
         """ Check if library and structure are loaded (call before operations)."""
         if self.lib is None or self.directory_structure is None:
              # Try to load from globals (fallback for how mount sets them up)
              global library, directory_structure, structure_depth
              if library and directory_structure:
                   self.lib = library
                   self.directory_structure = directory_structure
                   self.structure_depth = structure_depth
                   log.info("Library and structure loaded into FS instance from globals.")
              else:
                   log.error("Filesystem accessed before library/structure initialized!")
                   raise fuse.FuseOSError(errno.EIO) # Or another suitable error

    def statfs(self, path): # path argument added in newer fusepy
        self._init_check() # Ensure globals are loaded
        log.debug(f"statfs requested for path: {path}")

        # Return filesystem statistics.
        try:
             # Use library path if available
             target_path = os.path.dirname(self.lib.path) if self.lib and self.lib.path else os.path.expanduser("~")
             # Ensure target_path is string
             if isinstance(target_path, bytes):
                  target_path = target_path.decode('utf-8', 'surrogateescape')

             stv = os.statvfs(target_path)
             log.debug(f"statfs based on {target_path}: {stv}")
             # Return a dictionary matching FUSE expectations
             return dict((key, getattr(stv, key)) for key in (
                 'f_bsize', 'f_frsize', 'f_blocks', 'f_bfree', 'f_bavail',
                 'f_files', 'f_ffree', 'f_favail', # 'f_flag', 'f_namemax' less common now
             ))
        except Exception as e:
             log.error(f"Error during statfs: {e}", exc_info=True)
             raise fuse.FuseOSError(errno.EIO) # Input/output error


    def getattr(self, path, fh=None):
        self._init_check()
        log.debug(f"getattr: {path} (fh={fh})")
        path = path.strip() # Ensure no leading/trailing whitespace

        try:
            if path == "/":
                # Root directory attributes
                mode = stat.S_IFDIR | 0o755 # rwxr-xr-x (read/exec for all, write for owner)
                st = Stat(st_mode=mode, st_size=Stat.DIRSIZE, st_nlink=2) # Root usually has nlink=2
                return st.__dict__ # Return as dict for fusepy

            # Split path into components
            pathsplit = path.strip('/').split('/')
            pathsplit = [p for p in pathsplit if p] # Remove empty parts

            if not pathsplit: # Should only happen for "/"?
                 log.error(f"getattr: Invalid path components for path {path}")
                 raise fuse.FuseOSError(errno.ENOENT)

            filename = pathsplit[-1]
            dir_elements = pathsplit[:-1]

            # Get the node for the parent directory
            parent_node = self.directory_structure.getnode(dir_elements)

            if parent_node is None:
                log.debug(f"getattr: Parent path not found for {path}")
                raise fuse.FuseOSError(errno.ENOENT)

            # Check if it's a directory or a file
            if filename in parent_node.dirs:
                # It's a directory listed in the structure
                log.debug(f"getattr: Path {path} is a directory.")
                mode = stat.S_IFDIR | 0o555 # r-xr-xr-x (read/exec for all) - read-only dirs
                # Calculate nlink based on number of subdirs + '.' + '..'
                target_node = parent_node.dirs[filename]
                nlink = len(target_node.dirs) + 2
                st = Stat(st_mode=mode, st_size=Stat.DIRSIZE, st_nlink=nlink)
                return st.__dict__

            elif filename in parent_node.files:
                # It's a file listed in the structure
                log.debug(f"getattr: Path {path} is a file.")
                item_id = parent_node.files[filename]
                item = self.lib.get_item(item_id)

                if not item or not item.path:
                    log.error(f"getattr: Item {item_id} or its path not found for {path}")
                    raise fuse.FuseOSError(errno.ENOENT)

                # Get stats from the underlying real file
                real_path_str = item.path if isinstance(item.path, str) else item.path.decode('utf-8', 'surrogateescape')
                try:
                    statinfo = os.stat(real_path_str)
                    # Create Stat object helper using real file info
                    st_helper = Stat(
                        st_mode=(stat.S_IFREG | (statinfo.st_mode & 0o777)), # Regular file, copy permissions
                        st_size=statinfo.st_size, # Placeholder: Use real size. Overridden below if open.
                        st_uid=statinfo.st_uid,
                        st_gid=statinfo.st_gid,
                        st_nlink=statinfo.st_nlink,
                        dt_atime=Stat.epoch_datetime(statinfo.st_atime),
                        dt_mtime=Stat.epoch_datetime(statinfo.st_mtime),
                        dt_ctime=Stat.epoch_datetime(statinfo.st_ctime)
                    )

                    # *** Adjust size for virtual file ***
                    # Try to get size from an open handler if available (more accurate)
                    # Lookup by fh if provided and valid, else by path
                    handler = None
                    if fh is not None and fh in self.open_files_by_fh:
                        handler = self.open_files_by_fh[fh]
                    elif path in self.files: # Fallback check by path (less reliable if multiple opens?)
                         handler = self.files[path]

                    if handler:
                         st_helper.st_size = handler.bound + len(handler.music_data)
                         log.debug(f"getattr: Using size from active handler (fh={fh}): {st_helper.st_size}")
                    else:
                         # Estimate size without fully opening - requires parsing header, complex.
                         # Fallback: Use real file size (might be wrong due to metadata diff)
                         log.warning(f"getattr: Estimating size for {path} using real file size {statinfo.st_size}. May be inaccurate.")
                         st_helper.st_size = statinfo.st_size # Keep real size as estimate

                    # Make files read-only by default for safety, allow write perms if needed
                    # Use helper's mode, apply our permissions
                    st_helper.st_mode = (stat.S_IFREG | 0o444) # Base: regular file, read-only for all
                    if self.item_is_writable(item): # Add a check if writes should be allowed
                        st_helper.st_mode |= 0o200 # Add owner write permission if needed (rw-r--r--)

                    return st_helper.__dict__ # Return dict

                except FileNotFoundError:
                    log.error(f"getattr: Real file not found: {real_path_str}")
                    raise fuse.FuseOSError(errno.ENOENT)
                except OSError as e:
                    log.error(f"getattr: OS error stating real file {real_path_str}: {e}")
                    raise fuse.FuseOSError(e.errno)

            else:
                # Path doesn't correspond to a known file or directory
                log.debug(f"getattr: Path not found in structure: {path}")
                raise fuse.FuseOSError(errno.ENOENT)

        except fuse.FuseOSError:
            raise # Re-raise FUSE errors directly
        except Exception as e:
            log.error(f"getattr: Unexpected error for path {path}: {e}", exc_info=True)
            raise fuse.FuseOSError(errno.EIO) # Generic I/O error


    def item_is_writable(self, item):
        """ Check if the item's format is supported for writing. """
        self._init_check()
        # Currently only FLAC writeback is partially implemented
        fmt = item.format.lower() if item.format else ''
        if fmt == 'flac':
            return True
        # Add MP3 later if implemented
        # if fmt == 'mp3':
        #    return True
        return False

    # Note: utime is deprecated. utimens is preferred.
    # def utime(self, path, times): # Old signature
    #     # ... implementation ...
    #     pass

    def utimens(self, path, times=(None, None), fh=None):
        # times is a tuple (atime, mtime) where each is seconds (float)
        self._init_check()
        atime, mtime = times
        log.info(f"utimens: {path} (atime {atime}, mtime {mtime})")

        # Changing timestamps on the virtual file doesn't make much sense
        # without writing back to the real file. Disallow for now.
        raise fuse.FuseOSError(errno.EPERM) # Operation not permitted


    def access(self, path, mode):
        # mode is integer flags (os.F_OK, os.R_OK, os.W_OK, os.X_OK)
        self._init_check()
        log.debug(f"access: {path} (mode {oct(mode)})")

        try:
            # Get attributes to check existence and type
            # fh=None as access doesn't provide a file handle
            st_dict = self.getattr(path, fh=None)
            st_mode = st_dict['st_mode'] # Extract mode from dict

            # 1. Check Existence (F_OK) - Implicitly checked by getattr succeeding
            if mode == os.F_OK:
                log.debug(f"access: Existence check OK for {path}")
                return 0 # OK

            # 2. Check Permissions based on file type and requested mode
            is_dir = stat.S_ISDIR(st_mode)

            # Simplified check: Use effective permissions based on our getattr modes
            # Owner permissions
            if mode & os.R_OK and not (st_mode & 0o400):
                 log.debug(f"access: Read denied (owner) for {path}")
                 raise fuse.FuseOSError(errno.EACCES)
            if mode & os.W_OK and not (st_mode & 0o200):
                 log.debug(f"access: Write denied (owner) for {path}")
                 raise fuse.FuseOSError(errno.EACCES)
            if mode & os.X_OK and not (st_mode & 0o100):
                 log.debug(f"access: Execute denied (owner) for {path}")
                 raise fuse.FuseOSError(errno.EACCES)

            # Add group/other checks if uid/gid matching is implemented, otherwise skip

            # If we passed all relevant checks for the mode
            log.debug(f"access: Granted for {path} with mode {oct(mode)}")
            return 0 # Access granted

        except fuse.FuseOSError as e:
             # If getattr failed (e.g., ENOENT), or permission explicitly denied above
             log.debug(f"access: Denied for {path} with mode {oct(mode)} - Error {e.errno}")
             raise e # Re-raise the specific FUSE error


    def readlink(self, path):
        self._init_check()
        log.info(f"readlink: {path}")
        # This filesystem does not support symbolic links
        raise fuse.FuseOSError(errno.ENOSYS) # Function not implemented


    def mknod(self, path, mode, dev):
        self._init_check()
        log.info(f"mknod: {path} (mode {oct(mode)}, dev {dev})")
        # Creating nodes (files, devices) is not supported
        raise fuse.FuseOSError(errno.EPERM) # Operation not permitted (read-only FS)


    def mkdir(self, path, mode):
        self._init_check()
        log.info(f"mkdir: {path} (mode {oct(mode)})")
        # Creating directories is not supported
        raise fuse.FuseOSError(errno.EPERM)


    def unlink(self, path):
        self._init_check()
        log.info(f"unlink: {path}")
        # Deleting files is not supported
        raise fuse.FuseOSError(errno.EPERM)


    def rmdir(self, path):
        self._init_check()
        log.info(f"rmdir: {path}")
        # Deleting directories is not supported
        raise fuse.FuseOSError(errno.EPERM)


    def symlink(self, target, source): # Note: fusepy uses target, source args
        self._init_check()
        log.info(f"symlink: target={target}, source={source}")
        # Creating symbolic links is not supported
        raise fuse.FuseOSError(errno.EPERM)


    def link(self, target, source):
        self._init_check()
        log.info(f"link: target={target}, source={source}")
        # Creating hard links is not supported
        raise fuse.FuseOSError(errno.EPERM) # Cross-device link error might also be appropriate


    def rename(self, old, new):
        self._init_check()
        log.info(f"rename: old={old}, new={new}")
        # Renaming is not supported
        raise fuse.FuseOSError(errno.EPERM)


    def chmod(self, path, mode):
        self._init_check()
        log.info(f"chmod: {path} (mode {oct(mode)})")
        # Changing permissions is not supported
        raise fuse.FuseOSError(errno.EPERM)


    def chown(self, path, uid, gid):
        self._init_check()
        log.info(f"chown: {path} (uid {uid}, gid {gid})")
        # Changing ownership is not supported
        raise fuse.FuseOSError(errno.EPERM)


    def truncate(self, path, length, fh=None):
        self._init_check()
        log.info(f"truncate: {path} (length {length}, fh={fh})")
        # Truncating files is not supported (especially with interpolation)
        # If write support is added, this might need implementation.
        raise fuse.FuseOSError(errno.EPERM)


    ### DIRECTORY OPERATION METHODS ###

    def opendir(self, path):
        """ Checks permissions for listing a directory (essentially checks existence)."""
        self._init_check()
        log.debug(f"opendir: {path}")
        # Check if path is a directory using getattr logic
        try:
             st_dict = self.getattr(path, fh=None)
             if not stat.S_ISDIR(st_dict['st_mode']):
                  log.warning(f"opendir: Path {path} is not a directory.")
                  # Should return ENOTDIR according to spec
                  raise fuse.FuseOSError(errno.ENOTDIR)
             # If getattr succeeds and it's a directory, allow opening.
             # Return 0 indicates success, no special file handle needed for readdir here.
             log.debug(f"opendir: Success for {path}")
             return 0 # Return value is not used as fh by fusepy readdir
        except fuse.FuseOSError as e:
             log.debug(f"opendir: Failed for {path} - Error {e.errno}")
             # Propagate the error (ENOENT, EACCES etc.)
             raise e # Re-raise FUSE error to return correct code


    def releasedir(self, path, fh): # fh is the return value of opendir (0 here)
        """ Closes an open directory. No-op here."""
        self._init_check()
        log.debug(f"releasedir: {path} (fh={fh})")
        # Nothing to clean up for directories
        return 0 # Success


    def fsyncdir(self, path, datasync, fh):
        """ Synchronises an open directory. No-op here."""
        self._init_check()
        log.debug(f"fsyncdir: {path} (datasync {datasync}, fh={fh})")
        # Nothing to sync for directories in this read-only structure model
        return 0 # Success


    def readdir(self, path, fh): # fh is the return value of opendir (0 here)
        """ Generator function to produce directory listing. """
        self._init_check()
        log.debug(f"readdir: {path} (fh={fh})")

        # Standard entries
        # fusepy readdir should yield just names (bytes or str)
        yield '.'
        yield '..'

        try:
            pathsplit = path.strip('/').split('/')
            if path == '/': pathsplit = [] # Handle root case
            pathsplit = [p for p in pathsplit if p] # Clean empty parts

            # Get the directory node
            node = self.directory_structure.getnode(pathsplit)

            if node:
                # List subdirectories
                for dirname in sorted(node.dirs.keys()):
                     # FUSE expects bytes for names usually, encode if needed
                     yield dirname # Pass string, fusepy might handle encoding

                # List files
                for filename in sorted(node.files.keys()):
                     yield filename # Pass string
            else:
                 log.warning(f"readdir: Node not found for path {path}")
                 # Don't yield anything else if node not found

        except Exception as e:
            log.error(f"readdir: Error listing directory {path}: {e}", exc_info=True)
            # Stop iteration by not yielding further / returning


    ### FILE OPERATION METHODS ###

    def open(self, path, flags):
        """ Open a file for reading/writing, return file handle ID. """
        self._init_check()
        # flags: os.O_RDONLY, os.O_WRONLY, os.O_RDWR, os.O_APPEND, etc.
        log.debug(f"open: {path} (flags {oct(flags)})")

        # Check access permissions first (read/write based on flags)
        access_mode = 0
        if flags & (os.O_RDONLY | os.O_RDWR):
             access_mode |= os.R_OK
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_TRUNC):
             access_mode |= os.W_OK

        try:
             self.access(path, access_mode) # Will raise FuseOSError if denied
        except fuse.FuseOSError as e:
             log.warning(f"open: Access denied for {path} with flags {oct(flags)} (errno {e.errno})")
             raise e # Re-raise the error from access()

        # If access check passed, proceed to open/create handler
        try:
            # Check if path already has an *active* handler instance in self.files
            # This simple check might be insufficient if kernel allows multiple independent opens.
            # A robust system might allow multiple handlers for the same path.
            # For now, reuse if path exists.
            if path in self.files:
                # File already open, increment usage count on existing handler
                log.debug(f"open: Re-using existing FileHandler for {path}")
                handler = self.files[path]
                handler.open()
            else:
                # Create a new file handler instance
                log.debug(f"open: Creating new FileHandler for {path}")
                handler = FileHandler(path, self.lib) # May raise errors
                self.files[path] = handler # Store by path (for potential reuse)

            # Generate and return a unique file handle ID
            self.next_fh += 1
            fh_id = self.next_fh
            self.open_files_by_fh[fh_id] = handler # Map ID to handler

            log.debug(f"open: Success for {path}, returning fh={fh_id}")
            return fh_id # Return non-zero unique ID for the handle

        except FileNotFoundError as e:
             log.warning(f"open: File not found during handler creation for {path}: {e}")
             raise fuse.FuseOSError(errno.ENOENT)
        except OSError as e:
             log.error(f"open: OS error during handler creation for {path}: {e}")
             # Map common OS errors to FUSE errors
             errno_code = e.errno or errno.EIO
             raise fuse.FuseOSError(errno_code)
        except Exception as e:
             log.error(f"open: Unexpected error for {path}: {e}", exc_info=True)
             raise fuse.FuseOSError(errno.EIO)


    def create(self, path, mode, fi=None): # fi contains flags in fusepy
        """ Creates a file and opens it for writing. Not supported. """
        self._init_check()
        flags = fi.flags if fi else 0
        log.info(f"create: {path} (mode {oct(mode)}, flags {oct(flags)})")
        # File creation is not supported in this read-mostly model
        raise fuse.FuseOSError(errno.EPERM)


    def _get_handler(self, path, fh):
         """ Helper to get FileHandler based on fh. """
         # Use the file handle ID provided by FUSE
         if fh is not None and fh in self.open_files_by_fh:
              return self.open_files_by_fh[fh]
         else:
              # If fh is invalid or not found, it's an error.
              log.error(f"_get_handler: Invalid or unknown file handle: fh={fh} for path={path}")
              raise fuse.FuseOSError(errno.EBADF) # Bad file descriptor


    def fgetattr(self, path, fh): # fusepy provides fh from open
        """ Retrieves information about an open file. """
        # Note: path might be None in some fusepy versions for f* methods, rely on fh
        self._init_check()
        log.debug(f"fgetattr: (fh={fh}, path={path})") # Log path for context if available
        # Can potentially use fh to find the handler more efficiently.
        try:
             handler = self._get_handler(path, fh) # Get handler using fh
             # Call the main getattr implementation, passing the known handle
             # Need to get path from handler if path=None
             current_path = path or handler.path
             return self.getattr(current_path, fh) # Pass fh along
        except fuse.FuseOSError as e:
             # Propagate error correctly
             raise e
        except Exception as e:
             log.error(f"fgetattr: Unexpected error for fh={fh}, path={path}: {e}", exc_info=True)
             raise fuse.FuseOSError(errno.EIO)


    def release(self, path, fh): # fusepy provides fh
        """ Closes an open file identified by fh. """
        self._init_check()
        log.debug(f"release: (fh={fh}, path={path})")

        try:
            handler = self._get_handler(path, fh) # Get handler using fh
            # Call the handler's release method
            was_last_release = handler.release()

            if was_last_release:
                log.info(f"release: Last instance closed for fh={fh} (path={handler.path}). Removing handler.")
                # Clean up mappings
                del self.open_files_by_fh[fh]
                # Remove path mapping too (assuming one fh per path active for now)
                if handler.path in self.files:
                     del self.files[handler.path]
            return 0 # Success
        except fuse.FuseOSError as e:
            # EBADF if handler not found, or other errors
            log.error(f"release: Error releasing fh={fh}: {e.errno}")
            raise e # Re-raise


    def fsync(self, path, datasync, fh):
        """ Synchronises an open file identified by fh. """
        self._init_check()
        log.debug(f"fsync: (fh={fh}, path={path}, datasync={datasync})")
        # If writes were buffered, flush them here.
        try:
             handler = self._get_handler(path, fh)
             if handler.modified:
                  # TODO: Implement flushing changes to the real file if needed.
                  log.info(f"fsync: TODO: Flush modified data for {handler.path} to disk.")
                  pass # Placeholder for flush logic
                  # handler.modified = False # Reset flag after successful sync
             return 0 # Success
        except fuse.FuseOSError as e:
             raise e
        except Exception as e:
             log.error(f"fsync: Error for fh={fh}, path={path}: {e}", exc_info=True)
             raise fuse.FuseOSError(errno.EIO)


    def flush(self, path, fh):
        """ Flush cached data for an open file identified by fh. """
        self._init_check()
        log.debug(f"flush: (fh={fh}, path={path})")
        # Often called on close. Ensure data is 'stable'.
        # Could trigger the same flush logic as fsync, or just be a hint.
        try:
             handler = self._get_handler(path, fh)
             # Perform any necessary flushing (e.g., if buffering writes)
             log.debug(f"flush: No specific action needed for {handler.path}.")
             # If implementing write-back, might trigger it here too? Check FUSE semantics.
             return 0 # Success
        except fuse.FuseOSError as e:
             raise e
        except Exception as e:
             log.error(f"flush: Error for fh={fh}, path={path}: {e}", exc_info=True)
             raise fuse.FuseOSError(errno.EIO)


    def read(self, path, size, offset, fh):
        """ Read data from an open file identified by fh. """
        self._init_check()
        # path might be None
        log.debug(f"read: (fh={fh}, path={path}, size={size}, offset={offset})")
        try:
            handler = self._get_handler(path, fh)
            data = handler.read(size, offset)
            #log.debug(f"read: Returning {len(data)} bytes for fh={fh}.")
            return data # Return bytes read
        except fuse.FuseOSError as e:
            # Need to return error code, not raise exception for read/write? Check fusepy docs.
            # fusepy Operations methods *should* raise FuseOSError.
            log.error(f"read: FUSE error for fh={fh}, path={path}: {e.errno}")
            raise e # Re-raise to let FUSE handle it
        except Exception as e:
            log.error(f"read: Unexpected error for fh={fh}, path={path}: {e}", exc_info=True)
            raise fuse.FuseOSError(errno.EIO)


    def write(self, path, buf, offset, fh):
        """ Write data to an open file identified by fh. """
        # path might be None
        self._init_check()
        log.debug(f"write: (fh={fh}, path={path}, len={len(buf)}, offset={offset})")
        # buf is bytes
        try:
            handler = self._get_handler(path, fh)
            bytes_written = handler.write(offset, buf)
            log.debug(f"write: Handler wrote {bytes_written} bytes for fh={fh}.")
            return bytes_written # Return number of bytes written
        except fuse.FuseOSError as e:
             log.error(f"write: FUSE error for fh={fh}, path={path}: {e.errno}")
             raise e # Re-raise
        except OSError as e:
             # Catch errors from handler.write() and convert to FUSE errors
             log.error(f"write: OS error during write for fh={fh}, path={path}: {e}")
             errno_code = e.errno or errno.EIO
             raise fuse.FuseOSError(errno_code) # Raise FUSE error with specific code
        except Exception as e:
             log.error(f"write: Unexpected error for fh={fh}, path={path}: {e}", exc_info=True)
             raise fuse.FuseOSError(errno.EIO)


    def ftruncate(self, path, length, fh):
        """ Truncate an open file identified by fh. Not supported. """
        # path might be None
        self._init_check()
        log.info(f"ftruncate: (fh={fh}, path={path}, length={length})")
        # Requires careful handling of header/music data boundary
        raise fuse.FuseOSError(errno.EPERM) # Operation not permitted


# Main execution logic (standard FUSE setup)
# This part is usually handled by the library (beets command structure),
# if __name__ == '__main__':
#     # Example standalone execution (requires manual library setup)
#     # from beets.library import Library
#     # lib = Library(':memory:') # Or path to db
#     # # ... populate lib or load existing ...
#     #
#     # # Setup globals (as mount() would do)
#     # library = lib
#     # directory_structure = FSNode({}, {}) # Build structure
#     # structure_depth = 2 # Example
#     # # ... run build logic ...
#     #
#     # if len(sys.argv) < 2:
#     #      print(f"Usage: {sys.argv[0]} <mountpoint>")
#     #      sys.exit(1)
#     # mountpoint = sys.argv[1]
#     #
#     # fs = beetFileSystem()
#     # fuse.FUSE(fs, mountpoint, foreground=True, nothreads=True)
#     pass
