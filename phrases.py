from __future__ import print_function
import csv
import random
import sys
import copy
# from builtins import enumerate

import numpy as np

c_bitvec_size = 10
c_min_bits = 3
c_max_bits = 20
c_num_replicate_missing = 5
c_ham_winners_fraction = 8
c_num_iters = 300
c_num_skip_phrases_fraction = 0.1
c_num_dbs = 30
c_db_rnd_asex = 0.2
c_db_rnd_sex = 0.4 # after asex selection
c_db_num_flip_muts = 0.5 # multiplied by num items in db. 1.0 would mean that each item in the db, on average flips one bit

def load_order_freq_tbl(o_fn):
	freq_tbl, d_words, s_phrase_lens = [], dict(), set()
	try:
		with open(o_fn, 'rb') as o_fhr:
			o_csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, _, version_str, _, snum_orders = next(o_csvr)
			version = int(version_str)
			if version != 1:
				raise IOError
			for iorder in range(int(snum_orders)):
				row = next(o_csvr)
				l_co_oids = next(o_csvr)
				phrase = row[2:]
				s_phrase_lens.add(len(phrase))
				# freq_tbl[tuple(phrase)] = row[0]
				freq_tbl.append(phrase)
				for word in phrase:
					id = d_words.get(word, -1)
					if id == -1:
						d_words[word] = len(d_words)
	except IOError:
		return

	num_uniques = len(d_words)

	return freq_tbl, num_uniques, d_words, s_phrase_lens


# success_orders_freq = dict()
freq_tbl, num_uniques, d_words, s_phrase_lens = load_order_freq_tbl('orders_success.txt')
num_ham_winners = num_uniques / c_ham_winners_fraction

l_dbs = [np.random.choice(a=[0, 1], size=(num_uniques, c_bitvec_size)) for _ in range(c_num_dbs)]
d_phrase_sel_mats, d_lens = dict(), dict()
for ilen, phrase_len in enumerate(s_phrase_lens):
	num_input_bits = ((phrase_len-1) * c_bitvec_size) + (c_num_replicate_missing * phrase_len)
	sel_mat = []
	for ibit in range(c_bitvec_size):
		num_bits = random.randint(c_min_bits, c_max_bits)
		l_sels = []
		for isel in range(num_bits):
			l_sels.append(random.randint(0, num_input_bits-1))
		sel_mat.append([l_sels, random.randint(1, num_bits)])
	d_phrase_sel_mats[phrase_len] = sel_mat
	d_lens[phrase_len] = ilen

l_skip_phrases = [[] for _ in s_phrase_lens]
num_skip_phrases = 0
for phrase in freq_tbl:
	plen = len(phrase)
	for iskip in range(plen):
		if random.random() > c_num_skip_phrases_fraction:
			continue
		l_skip_phrases[d_lens[plen]].append([phrase[:iskip] + phrase[iskip+1:], iskip, phrase[iskip]])
		num_skip_phrases += 1

for iiter in range(c_num_iters):
	l_scores = []
	min_score, max_score = sys.float_info.max, -sys.float_info.max
	for idb in range(c_num_dbs):
		num_hits = 0
		# nd_bit_db = np.random.choice(a=[0, 1], size=(num_uniques, c_bitvec_size))
		nd_bit_db = l_dbs[idb]
		for ilen, phrase_len in enumerate(s_phrase_lens):
			sel_mat = d_phrase_sel_mats[phrase_len]
			for skip_phrase in l_skip_phrases[ilen]:
				nd_bits = np.zeros(c_bitvec_size, dtype=np.int)
				input_bits = np.zeros(((phrase_len-1) * c_bitvec_size) + (c_num_replicate_missing * phrase_len), dtype=np.int)
				loc = 0
				for word in skip_phrase[0]:
					input_bits[loc:loc+c_bitvec_size] = nd_bit_db[d_words[word]]
					loc += c_bitvec_size
				missing_loc = skip_phrase[1] * c_num_replicate_missing
				input_bits[loc + missing_loc:loc + missing_loc + c_num_replicate_missing] = [1] * c_num_replicate_missing

				for iobit in range(c_bitvec_size):
					sum = 0
					for iibit in sel_mat[iobit][0]:
						sum += input_bits[iibit]
					nd_bits[iobit] = 1 if sum >= sel_mat[iobit][1] else 0

				hd = np.sum(np.absolute(np.subtract(nd_bits, nd_bit_db)), axis=1)
				hd_winners = np.argpartition(hd, (num_ham_winners + 1))[:(num_ham_winners + 1)]
				if d_words[skip_phrase[2]] in hd_winners:
					num_hits += 1

		score = float(num_hits)/float(num_skip_phrases)
		# print('score:', score)
		l_scores.append(score)
		if score > max_score:
			max_score = score
		if score < min_score:
			min_score = score

	print('avg score:', np.mean(l_scores)) # , 'list', l_scores)
	mid_score = (max_score + min_score) / 2.0
	range_scores = (max_score - mid_score)
	l_w_scores = np.array([(score - mid_score) / range_scores for score in l_scores])
	l_w_scores = np.where(l_w_scores > 0.0, l_w_scores, np.zeros_like(l_w_scores))
	sel_prob = l_w_scores/np.sum(l_w_scores)

	num_flip_muts = int(c_db_num_flip_muts * num_uniques)

	l_sel_dbs = np.random.choice(c_num_dbs, size=c_num_dbs, p=sel_prob)
	l_dbs = [copy.deepcopy(l_dbs[isel]) for isel in l_sel_dbs]
	for idb, nd_bit_db in enumerate(l_dbs):
		if random.random() < c_db_rnd_asex:
			for imut in range(num_flip_muts):
				allele, target = random.randint(0, c_bitvec_size - 1), random.randint(0, num_uniques-1)
				nd_bit_db[target][allele] = 1 if (nd_bit_db[target][allele] == 0) else 0
		elif random.random() < c_db_rnd_sex:
			partner_db = copy.deepcopy(random.choice(l_dbs))  # not the numpy function
			for allele in range(c_bitvec_size):
				for iun in range(num_uniques):
					if random.random() < 0.5:
						nd_bit_db[iun,:] = partner_db[iun,:]

print('done')

