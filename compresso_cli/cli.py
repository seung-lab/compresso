import os
import os.path
import sys

import click
import compresso
import numpy as np

@click.command()
@click.option("-c/-d", "--compress/--decompress", default=True, is_flag=True, help="Decompress to a numpy .npy file.", show_default=True)
@click.argument("source", nargs=-1)
def main(compress, source):
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
	for i in range(len(source)):
		if source[i] == "-":
			source = source[:i] + sys.stdin.readlines() + source[i+1:]
	
	for src in source:
		if compress:
			compress_file(src)
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

def compress_file(src):
	try:
		data = np.load(src)
	except ValueError:
		print(f"compresso: {src} is not a numpy file.")
		sys.exit()

	binary = compresso.compress(data)
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

