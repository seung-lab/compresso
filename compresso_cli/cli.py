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
@click.option("-c/-d", "--compress/--decompress", default=True, is_flag=True, help="Decompress to a numpy .npy file.", show_default=True)
@click.option('--steps', type=Tuple3(), default=None, help="Compression step size. No effect on decompression.", show_default=True)
@click.option('--six', is_flag=True, default=False, help="Use 6-way CCL instead of 4-way. No effect on decompression.", show_default=True)
@click.argument("source", nargs=-1)
def main(compress, source, steps, six):
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
		if compress:
			compress_file(src, steps, six)
		else:
			decompress_file(src)

def decompress_file(src):
	with open(src, "rb") as f:
		binary = f.read()

	try:
		data = compresso.decompress(binary)
	except compresso.DecodeError:
		print(f"compresso: {src} could not be decoded.")
		sys.exit()

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

def compress_file(src, steps, six):
	try:
		data = np.load(src)
	except ValueError:
		print(f"compresso: {src} is not a numpy file.")
		sys.exit()

	connectivity = 4 
	if six:
		connectivity = 6

	binary = compresso.compress(data, steps=steps, connectivity=connectivity)
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

