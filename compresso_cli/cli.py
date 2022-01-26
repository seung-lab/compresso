import os
import os.path
import sys

import click
import compresso
import numpy as np

class Tuple3(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple3'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
      if len(value) != 3:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
    return value
  

@click.command()
@click.option("-c/-d", "--compress/--decompress", default=True, is_flag=True, help="Compress from or decompress to a numpy .npy file.", show_default=True)
@click.option('-i', "--info", default=False, is_flag=True, help="Print the header for the file.", show_default=True)
@click.option('--steps', type=Tuple3(), default=None, help="Compression step size. No effect on decompression.", show_default=True)
@click.option('--six', is_flag=True, default=False, help="Use 6-way CCL instead of 4-way. No effect on decompression.", show_default=True)
@click.option('--z-index/--no-z-index', is_flag=True, default=True, help="Write a stream that has random access to z slices.", show_default=True)
@click.argument("source", nargs=-1)
def main(compress, info, source, steps, six, z_index):
	"""
	Compress and decompress compresso files to and from numpy .npy files.

	Compatible with compresso format version 0 streams.

	The compresso algorithm can be found in this paper:
	
	B. Matejek, D. Haehn, F. Lekschas, M. Mitzenmacher, and H. Pfister.
	"Compresso: Efficient Compression of Segmentation Data for Connectomics".
	Springer: Intl. Conf. on Medical Image Computing and Computer-Assisted Intervention.
	2017.

	This program was written by Brian Matejek and William Silversmith
	and is MIT licensed.
	"""
	if steps is not None and len(steps) != 3:
		raise ValueError(f"steps must be a comma delimited set of three numbers that multiply to > 0 and <= 64.")

	for i in range(len(source)):
		if source[i] == "-":
			source = source[:i] + sys.stdin.readlines() + source[i+1:]
	
	for src in source:
		if info:
			print_header(src)
			continue

		if compress:
			compress_file(src, steps, six, z_index)
		else:
			decompress_file(src)

def print_header(src):
	try:
		with open(src, "rb") as f:
			binary = f.read()
	except FileNotFoundError:
		print(f"compresso: File \"{src}\" does not exist.")
		return

	head = compresso.header(binary)
	print(f"Filename: {src}")
	for key,val in head.items():
		print(f"{key}: {val}")
	print()

def decompress_file(src):
	try:
		with open(src, "rb") as f:
			binary = f.read()
	except FileNotFoundError:
		print(f"compresso: File \"{src}\" does not exist.")
		return

	try:
		data = compresso.decompress(binary)
	except compresso.DecodeError:
		print(f"compresso: {src} could not be decoded.")
		return

	del binary

	dest = src.replace(".cpso", "")
	_, ext = os.path.splitext(dest)
	
	if ext != ".npy":
		dest += ".npy"

	np.save(dest, data)

	try:
		stat = os.stat(dest)
		if stat.st_size > 0:
			os.remove(src)
		else:
			raise ValueError("File is zero length.")
	except (FileNotFoundError, ValueError) as err:
		print(f"compresso: Unable to write {dest}. Aborting.")
		sys.exit()

def compress_file(src, steps, six, z_index):
	try:
		data = np.load(src)
	except ValueError:
		print(f"compresso: {src} is not a numpy file.")
		return
	except FileNotFoundError:
		print(f"compresso: File \"{src}\" does not exist.")
		return

	connectivity = 4 
	if six:
		connectivity = 6

	binary = compresso.compress(
		data, steps=steps, connectivity=connectivity,
		random_access_z_index=z_index
	)
	del data

	dest = f"{src}.cpso"
	with open(dest, "wb") as f:
		f.write(binary)
	del binary

	try:
		stat = os.stat(dest)
		if stat.st_size > 0:
			os.remove(src)
		else:
			raise ValueError("File is zero length.")
	except (FileNotFoundError, ValueError) as err:
		print(f"compresso: Unable to write {dest}. Aborting.")
		sys.exit()

