#!/usr/bin/env python3

import numpy as np
from argparse import ArgumentParser
from csv import DictWriter
from os import fstat
from sklearn.linear_model import LinearRegression
from tqdm import tqdm, trange

# Argparse initializations

argument_parser = ArgumentParser(description = 'Molecular Dynamics Mean Squared Distance and Diffusion Coefficient')
argument_parser.add_argument('data_file', type = str, help = 'Data file')
argument_parser.add_argument('dump_file', type = str, help = 'Dump file')
argument_parser.add_argument('adsorbent_atom_id_start', type = int, help = 'Adsorbent atom id start (inclusive)')
argument_parser.add_argument('adsorbent_atom_id_end', type = int, help = 'Adsorbent atom id end (inclusive)')
argument_parser.add_argument('adsorbate_atom_id_start', type = int, help = 'Adsorbate atom id start (inclusive)')
argument_parser.add_argument('adsorbate_atom_id_end', type = int, help = 'Adsorbate atom id end (inclusive)')
argument_parser.add_argument('layer_size', type = float, help = 'Size of a layer from the adsorbent\'s surface')
argument_parser.add_argument('--show_graph', action = 'store_true', help = 'Shows the MSD vs timesteps graph for your quick reference')

args = argument_parser.parse_args()
data_file = args.data_file
dump_file = args.dump_file
adsorbent_atom_id_start = args.adsorbent_atom_id_start
adsorbent_atom_id_end = args.adsorbent_atom_id_end
adsorbate_atom_id_start = args.adsorbate_atom_id_start
adsorbate_atom_id_end = args.adsorbate_atom_id_end
layer_size = args.layer_size
show_graph = args.show_graph

# Some helper functions

def is_adsorbent_atom(atom_id):
	return atom_id >= adsorbent_atom_id_start and atom_id <= adsorbent_atom_id_end

def is_adsorbate_atom(atom_id):
	return atom_id >= adsorbate_atom_id_start and atom_id <= adsorbate_atom_id_end

def squared_distance(coords1, coords2):
	dist = 0
	for i in range(3):
		dist += (coords1[i] - coords2[i]) ** 2
	return dist

def layer_index(squared_dist, layer_size):
	# Binary search over the answer
	l = 0
	r = 1000000000000000000
	while r - l != 1:
		mid = (l + r) // 2
		if squared_dist >= (mid * layer_size) ** 2:
			l = mid
		else:
			r = mid
	return l


# Helper class to find average coordinates of a set of molecules

class AverageCoords:

	def __init__(self):
		self.coords = [0, 0, 0]
		self.num = 0
	
	def add_contribution(self, coords):
		for i in range(len(self.coords)):
			self.coords[i] = (self.coords[i] * self.num + coords[i]) / (self.num + 1)
		self.num += 1

# Initializations using the data file

atom_id_to_mol_id = {}

with open(data_file, newline = '') as datafile:
	for _ in range(2):
		datafile.readline()
	num_atoms, _ = datafile.readline().split()
	num_atoms = int(num_atoms)
	for line in datafile:
		if line == 'Atoms\n':
			break
	datafile.readline()
	for _ in trange(num_atoms, desc = 'Processing data file'):
		line = datafile.readline()
		if line == '\n':
			break
		atom_id, mol_id, _, _, _, _, _ = line.split()
		atom_id = int(atom_id)
		mol_id = int(mol_id)
		atom_id_to_mol_id[atom_id] = mol_id

# Initializations using the dump file

time_deltas = []
mean_msds = []
initial_adsorbate_mols_avg_coords = {}
layer_indices = {}
num_layer_indices = -1

with open(dump_file, newline = '') as dumpfile, tqdm(total = fstat(dumpfile.fileno()).st_size, desc = 'Processing dump file') as pbar:
	while dumpfile.readline():
		time_deltas.append(int(dumpfile.readline().strip()))
		
		dumpfile.readline()
		
		num_atoms = int(dumpfile.readline().strip())

		for _ in range(5):
			dumpfile.readline()

		adsorbate_mols_avg_coords = {}
		if len(time_deltas) == 1:
			adsorbent_avg_coords = AverageCoords()

		for i in range(num_atoms):
			coords = [0] * 3
			atom_id, _, coords[0], coords[1], coords[2], _, _, _ = dumpfile.readline().split()
			atom_id = int(atom_id)
			for j in range(3):
				coords[j] = float(coords[j])
			
			if len(time_deltas) == 1 and is_adsorbent_atom(atom_id):
				adsorbent_avg_coords.add_contribution(coords)
			
			if is_adsorbate_atom(atom_id):
				mol_id = atom_id_to_mol_id[atom_id]
				if mol_id not in adsorbate_mols_avg_coords:
					adsorbate_mols_avg_coords[mol_id] = AverageCoords()
				adsorbate_mols_avg_coords[mol_id].add_contribution(coords)
		
		if len(time_deltas) == 1:
			initial_adsorbate_mols_avg_coords = adsorbate_mols_avg_coords.copy()
			for adsorbate_mol_id, adsorbate_avg_coords in adsorbate_mols_avg_coords.items():
				layer_indices[adsorbate_mol_id] = layer_index(squared_distance(adsorbent_avg_coords.coords, adsorbate_avg_coords.coords), layer_size)
			num_layer_indices = max(layer_indices.values()) + 1
			mean_msds = [[] for _ in range(num_layer_indices)]
		
		msds = [0] * num_layer_indices
		for adsorbate_mol_id, adsorbate_avg_coords in adsorbate_mols_avg_coords.items():
			msds[layer_indices[adsorbate_mol_id]] += squared_distance(initial_adsorbate_mols_avg_coords[adsorbate_mol_id].coords, adsorbate_avg_coords.coords)
		for l_index in range(len(msds)):
			msds[l_index] /= len(adsorbate_mols_avg_coords.items())
			mean_msds[l_index].append(msds[l_index])
		
		pbar.update(dumpfile.tell() - pbar.n)

for t_index in range(len(time_deltas) - 1, -1, -1):
	time_deltas[t_index] -= time_deltas[0]

with open('out.csv', 'w', newline = '') as csvfile:
	fieldnames = ['Layer index', 'Timestep', 'Mean MSD']
	writer = DictWriter(csvfile, fieldnames = fieldnames)
	writer.writeheader()
	for l_index in range(len(mean_msds)):
		for t_index in range(len(time_deltas)):
			writer.writerow({'Layer index' : l_index + 1, 'Timestep' : time_deltas[t_index], 'Mean MSD' : mean_msds[l_index][t_index]})

time_deltas = np.array(time_deltas).reshape(-1, 1)

if show_graph:
	import matplotlib.pyplot as plt
	for l_index in range(len(mean_msds)):
		plt.plot(time_deltas, mean_msds[l_index], label = f'Layer {l_index + 1}')
	plt.xlabel('Timesteps')
	plt.ylabel('Mean Squared Distance (MSD)')
	plt.legend()
	plt.show()

print('Calculated diffusion coefficients:')
for l_index in range(num_layer_indices):
	print(f'Layer {l_index + 1}: ', LinearRegression(fit_intercept = False).fit(time_deltas, np.array(mean_msds[l_index]).reshape(-1, 1)).coef_[0, 0] / 6)
