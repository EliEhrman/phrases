"""
Decscended from phrases2.py

This module attempts to break the els in the phrases into words and learn the initial bitvec dictionary that way
The goal is also to skip not just words but also groups of words
At the time of writing this comment, all that has happened is that the word breakdown seems to work
"""

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
c_num_iters = 30000
c_num_skip_phrases_fraction = 0.1
c_num_dbs = 7
c_num_sel_mats = 9
c_db_rnd_asex = 0.2
c_db_rnd_sex = 0.4 # after asex selection
c_db_num_flip_muts = 0.5 # multiplied by num items in db. 1.0 would mean that each item in the db, on average flips one bit
c_mid_score = c_num_dbs * 5 / 8
c_rnd_asex = 0.3
c_rnd_sex = 0.4 # after asex selection
c_num_incr_muts = 2
c_num_change_muts = 3
c_change_mut_prob_change_len = 0.3 # 0.3
c_change_mut_num_change = 1

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
				phrase = [el.split(' ') for el in phrase]
				phrase = [word for el in phrase for word in el]
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


def create_input_bits(nd_bit_db, d_words, skip_phrase):
	phrase_len = len(skip_phrase[0]) + 1
	input_bits = np.zeros(((phrase_len - 1) * c_bitvec_size) + (c_num_replicate_missing * phrase_len), dtype=np.int)
	loc = 0
	for word in skip_phrase[0]:
		input_bits[loc:loc + c_bitvec_size] = nd_bit_db[d_words[word]]
		loc += c_bitvec_size
	missing_loc = skip_phrase[1] * c_num_replicate_missing
	input_bits[loc + missing_loc:loc + missing_loc + c_num_replicate_missing] = [1] * c_num_replicate_missing
	return input_bits


def create_output_bits(sel_mat, input_bits):
	nd_bits = np.zeros(c_bitvec_size, dtype=np.int)

	for iobit in range(c_bitvec_size):
		sum = 0
		for iibit in sel_mat[iobit][0]:
			sum += input_bits[iibit]
		nd_bits[iobit] = 1 if sum >= sel_mat[iobit][1] else 0

	return nd_bits

def score_db_and_sel_mat(s_phrase_lens, d_words, l_skip_phrases, nd_bit_db, d_phrase_sel_mats):
	num_ham_winners = len(d_words) / c_ham_winners_fraction
	num_skip_phrases = 0
	num_hits = 0
	for ilen, phrase_len in enumerate(s_phrase_lens):
		num_skip_phrases += len(l_skip_phrases[ilen])
		sel_mat = d_phrase_sel_mats[phrase_len]
		for skip_phrase in l_skip_phrases[ilen]:
			# nd_bits = np.zeros(c_bitvec_size, dtype=np.int)
			input_bits = create_input_bits(nd_bit_db, d_words, skip_phrase)
			nd_bits = create_output_bits(sel_mat, input_bits)

			hd = np.sum(np.absolute(np.subtract(nd_bits, nd_bit_db)), axis=1)
			hd_winners = np.argpartition(hd, (num_ham_winners + 1))[:(num_ham_winners + 1)]
			if d_words[skip_phrase[2]] in hd_winners:
				num_hits += 1

	return float(num_hits) / float(num_skip_phrases)

def select_best(s_phrase_lens, d_words, l_skip_phrases, l_objs, iiter, l_record_scores, l_record_objs, best_other, b_do_dbs):
	min_score, max_score = sys.float_info.max, -sys.float_info.max
	num_objs = len(l_objs)
	l_scores = []
	for iobj in range(num_objs):
		if b_do_dbs:
			nd_bit_db = l_objs[iobj]
			d_phrase_sel_mats = best_other
		else:
			nd_bit_db = best_other
			d_phrase_sel_mats = l_objs[iobj]
		score = score_db_and_sel_mat(s_phrase_lens, d_words, l_skip_phrases, nd_bit_db, d_phrase_sel_mats)
		l_scores.append(score)
		if score > max_score:
			max_score = score
		if score < min_score:
			min_score = score

	# print('avg score:', np.mean(l_scores)) # , 'list', l_scores)
	print('iiter', iiter, 'avg score:', np.mean(l_scores), 'max score:', np.max(l_scores)) # , 'list', l_scores)
	if l_record_scores == [] or max_score > l_record_scores[0]:
		l_record_scores.insert(0, max_score)
		l_record_objs.insert(0, l_objs[l_scores.index(max_score)])
	else:
		l_objs[l_scores.index(min_score)] = l_record_objs[0]
		l_scores[l_scores.index(min_score)] = l_record_scores[0]
	# mid_score = (max_score + min_score) / 2.0
	mid_score = l_scores[np.array(l_scores).argsort()[c_mid_score]]
	if max_score == min_score:
			range_scores = max_score
			l_obj_scores = np.ones(len(l_scores), dtype=np.float32)
	elif mid_score == max_score:
		range_scores = max_score - min_score
		l_obj_scores = np.array([(score - min_score) / range_scores for score in l_scores])
	else:
		range_scores = max_score - mid_score
		l_obj_scores = np.array([(score - mid_score) / range_scores for score in l_scores])
	l_obj_scores = np.where(l_obj_scores > 0.0, l_obj_scores, np.zeros_like(l_obj_scores))
	sel_prob = l_obj_scores/np.sum(l_obj_scores)
	l_sel_dbs = np.random.choice(num_objs, size=num_objs, p=sel_prob)
	l_objs[:] = [copy.deepcopy(l_objs[isel]) for isel in l_sel_dbs]

def mutate_dbs(l_dbs, num_uniques):
	num_flip_muts = int(c_db_num_flip_muts * num_uniques)

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

def mutate_sel_mats(l_d_phrase_sel_mats, s_phrase_lens):
	for isel, d_phrase_sel_mats in enumerate(l_d_phrase_sel_mats):
		for ilen, phrase_len in enumerate(s_phrase_lens):
			num_input_bits = ((phrase_len - 1) * c_bitvec_size) + (c_num_replicate_missing * phrase_len)
			sel_mat = d_phrase_sel_mats[phrase_len]
			if random.random() < c_rnd_asex:
				for imut in range(c_num_incr_muts):
					allele = random.randint(0, c_bitvec_size-1)
					num_bits = len(sel_mat[allele][0])
					if sel_mat[allele][1] < num_bits-2:
						sel_mat[allele][1] += 1
				for imut in range(c_num_incr_muts):
					allele = random.randint(0, c_bitvec_size-1)
					if sel_mat[allele][1] > 1:
						sel_mat[allele][1] -= 1
				for icmut in range(c_num_change_muts):
					allele = random.randint(0, c_bitvec_size-1)
					bit_list = sel_mat[allele][0]
					if random.random() < c_change_mut_prob_change_len:
						if len(bit_list) < c_max_bits:
							bit_list.append(random.randint(0, num_input_bits - 1))
					elif random.random() < c_change_mut_prob_change_len:
						if len(bit_list) > c_min_bits:
							bit_list.pop(random.randrange(len(bit_list)))
							if sel_mat[allele][1] >= len(bit_list) - 1:
								sel_mat[allele][1] -= 1
					else:
						for ichange in range(c_change_mut_num_change):
							bit_list[random.randint(0, len(bit_list)-1)] = random.randint(0, num_input_bits - 1)
			elif random.random() < c_rnd_sex:
				partner_sel_mat = copy.deepcopy(random.choice(l_d_phrase_sel_mats)[phrase_len]) # not the numpy function
				for allele in range(c_bitvec_size):
					if random.random() < 0.5:
						sel_mat[allele] = list(partner_sel_mat[allele])

def main():
	# success_orders_freq = dict()
	freq_tbl, num_uniques, d_words, s_phrase_lens = load_order_freq_tbl('orders_success.txt')

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


	l_d_phrase_sel_mats = [copy.deepcopy(d_phrase_sel_mats) for isel in range(c_num_sel_mats)]
	l_record_sel_mat_scores = [-1.0]
	l_record_sel_mats = [d_phrase_sel_mats]
	l_record_db_scores = []
	l_record_dbs = []

	for iiter in range(c_num_iters):
		# select_best(l_objs, iiter, l_record_scores, l_record_objs, best_other, b_do_dbs)
		if iiter % 2 == 0:
			# if iiter == 0:
			select_best(s_phrase_lens, d_words, l_skip_phrases, l_dbs, iiter,
						l_record_db_scores, l_record_dbs, l_record_sel_mats[0], b_do_dbs=True)
			mutate_dbs(l_dbs, num_uniques)
		else:
			select_best(s_phrase_lens, d_words, l_skip_phrases, l_d_phrase_sel_mats, iiter,
						l_record_sel_mat_scores, l_record_sel_mats, l_record_dbs[0], b_do_dbs=False)
			mutate_sel_mats(l_d_phrase_sel_mats, s_phrase_lens)

main()
print('done')

