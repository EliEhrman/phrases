"""
Descended from phrases.py

THis module seeks to add to the dictionary bitvec as new words come in

The skip representation will be replaced by a mask on the input for the unknown word

"""
from __future__ import print_function
import csv
import random
import sys
import copy
from os.path import expanduser
# from builtins import enumerate

import numpy as np

# fnt = 'orders_success.txt'
fnt = '~/tmp/adv_phrase_freq.txt'

c_bitvec_size = 11
c_min_bits = 3
c_max_bits = 20
c_num_replicate_missing = 5
c_ham_winners_fraction = 32
c_num_iters = 300000
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
c_init_len = 800

c_move_rnd = 0.5
c_move_rnd_change = 0.02
c_min_frctn_change  = 0.001
c_max_frctn_change  = 0.01

def create_word_dict(phrase_list, max_process):
	d_els = dict()
	for iphrase, phrase in enumerate(phrase_list):
		if iphrase > max_process:
			break
		for iel, el, in enumerate(phrase):
			for word in phrase:
				id = d_els.get(word, -1)
				if id == -1:
					d_els[word] = len(d_els)

	return d_els


def load_order_freq_tbl(fnt):
	fn = expanduser(fnt)

	freq_tbl, d_words, s_phrase_lens = [], dict(), set()
	try:
		with open(fn, 'rb') as o_fhr:
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
				# for word in phrase:
				# 	id = d_words.get(word, -1)
				# 	if id == -1:
				# 		d_words[word] = len(d_words)
	except IOError:
		raise ValueError('Cannot open or read ', fn)

	# num_uniques = len(d_words)

	# return freq_tbl, num_uniques, d_words, s_phrase_lens
	random.shuffle(freq_tbl)
	return freq_tbl, s_phrase_lens


def create_input_bits(nd_bit_db, d_words, phrase):
	phrase_len = len(phrase)
	input_bits = np.zeros(phrase_len * c_bitvec_size, dtype=np.uint8)
	loc = 0
	for word in phrase:
		input_bits[loc:loc + c_bitvec_size] = nd_bit_db[d_words[word]]
		loc += c_bitvec_size
	# missing_loc = phrase[1] * c_num_replicate_missing
	# input_bits[loc + missing_loc:loc + missing_loc + c_num_replicate_missing] = [1] * c_num_replicate_missing
	return np.array(input_bits)


def create_output_bits(sel_mat, input_bits):
	nd_bits = np.zeros(c_bitvec_size, dtype=np.int)

	for iobit in range(c_bitvec_size):
		sum = 0
		for iibit in sel_mat[iobit][0]:
			sum += input_bits[iibit]
		nd_bits[iobit] = 1 if sum >= sel_mat[iobit][1] else 0

	return nd_bits


def score_hd_output_bits(nd_phrase_bits_db, qbits, mbits, iskip, iword, change_db):
	numrecs = nd_phrase_bits_db.shape[0]

	def calc_score(outputs):
		odiffs = np.logical_and(np.not_equal(qbits, outputs), np.logical_not(mbits))
		nd_diffs = np.where(odiffs, np.ones_like(outputs), np.zeros_like(outputs))
		divider = np.array(range(1, nd_diffs.shape[0] + 1), np.float32)
		return np.sum(np.divide(np.sum(nd_diffs, axis=1), divider))

	# nd_diffs = np.absolute(np.subtract(qbits, nd_phrase_bits_db))
	nd_diffs = np.logical_and(np.not_equal(qbits, nd_phrase_bits_db), mbits)
	nd_diffs = np.where(nd_diffs, np.ones_like(nd_phrase_bits_db), np.zeros_like(nd_phrase_bits_db))
	hd = np.sum(nd_diffs, axis=1)
	hd_winners = np.argpartition(hd, (score_hd_output_bits.num_ham_winners + 1))[:(score_hd_output_bits.num_ham_winners + 1)]
	hd_of_winners = hd[hd_winners]
	iwinners = np.argsort(hd_of_winners)
	hd_idx_sorted = hd_winners[iwinners]
	winner_outputs = nd_phrase_bits_db[hd_idx_sorted]
	avg_outputs = nd_phrase_bits_db[np.random.randint(numrecs, size=hd_idx_sorted.shape[0])]
	obits = winner_outputs[:, iskip*c_bitvec_size:(iskip+1)*c_bitvec_size]
	bad_obits = avg_outputs[:, iskip*c_bitvec_size:(iskip+1)*c_bitvec_size]
	# ibits = qbits[iskip*c_bitvec_size:(iskip+1)*c_bitvec_size].astype(float)
	# obits_goal = np.where(np.average(obits, axis=0) > 0.5, np.ones_like(ibits), np.zeros_like(ibits))
	obits_goal, obits_keep_away = np.average(obits, axis=0), np.average(bad_obits, axis=0)
	new_obits_goal = ((obits_goal + (np.ones(c_bitvec_size) - obits_keep_away)) / 2.0).tolist()
	if change_db[iword][1] == 0.0:
		change_db[iword][0] = new_obits_goal
	else:
		change_db[iword][0] = ((np.array(change_db[iword][0]) * change_db[iword][1]) + new_obits_goal) / (change_db[iword][1] + 1.0)
	change_db[iword][1] += 1.0
	close_score, avg_score = calc_score(winner_outputs), calc_score(avg_outputs)
	return avg_score / (close_score + 10.0)


score_hd_output_bits.num_ham_winners = 0


def score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, l_change_db):
	num_uniques = len(d_words)
	l_change_db = [[[0.0 for ibit in xrange(c_bitvec_size)], 0.0] for _ in xrange(num_uniques)]
	num_ham_winners = len(d_words) / c_ham_winners_fraction
	score_hd_output_bits.num_ham_winners= num_ham_winners
	bitvec_size = nd_bit_db.shape[1]
	num_phrases = 0
	num_hits = 0
	l_l_mbits = [] # mask bits
	for ilen, phrase_len in enumerate(s_phrase_lens):
		l_mbits = []
		for iskip in range(phrase_len):
			mbits = np.ones(phrase_len * bitvec_size, np.uint8)
			mbits[iskip*bitvec_size:(iskip+1)*bitvec_size] = np.zeros(bitvec_size, np.uint8)
			l_mbits.append(mbits)
		l_l_mbits.append(l_mbits)

	phrase_bits_db = [np.zeros((len(l_len_phrases), bitvec_size * list(s_phrase_lens)[ilen]), dtype=np.int)
					  for ilen, l_len_phrases in enumerate(l_phrases)]
	score = 0.0
	for ilen, phrase_len in enumerate(s_phrase_lens):
		num_phrases += len(l_phrases[ilen])
		# sel_mat = d_phrase_sel_mats[phrase_len]
		for iphrase, phrase in enumerate(l_phrases[ilen]):
			# nd_bits = np.zeros(c_bitvec_size, dtype=np.int)
			input_bits = create_input_bits(nd_bit_db, d_words, phrase)
			phrase_bits_db[ilen][iphrase, :] = input_bits

		for iphrase, phrase in enumerate(l_phrases[ilen]):
			for iskip in range(phrase_len):
				score += score_hd_output_bits(	phrase_bits_db[ilen], phrase_bits_db[ilen][iphrase],
												l_l_mbits[ilen][iskip], iskip, d_words[phrase[iskip]],
												l_change_db)

	num_changed = 0
	for iuniqe, bits_data in enumerate(l_change_db):
		l_bits_avg, _ = bits_data
		l_bits_now = nd_bit_db[iuniqe]
		if random.random() < score_and_change_db.move_rnd:
			ibit = random.randint(0, c_bitvec_size-1)
			bit_now, bit_goal = l_bits_now[ibit], l_bits_avg[ibit]
			if bit_now == 0 and bit_goal > 0.5:
				if random.random() < (bit_goal - 0.5):
					nd_bit_db[iuniqe][ibit] = 1
					num_changed += 1
			elif bit_now == 1 and bit_goal < 0.5:
				if random.random() < (0.5 - bit_goal):
					nd_bit_db[iuniqe][ibit] = 0
					num_changed += 1
	frctn_change = float(num_changed) / float(num_uniques * c_bitvec_size)
	if frctn_change < c_min_frctn_change:
		score_and_change_db.move_rnd += c_move_rnd_change
	elif frctn_change > c_max_frctn_change:
		score_and_change_db.move_rnd -= c_move_rnd_change
	print(num_changed, 'bits changed out of', num_uniques * c_bitvec_size, 'fraction:',
		  frctn_change, 'move_rnd = ', score_and_change_db.move_rnd)
	return score



score_and_change_db.move_rnd = c_move_rnd

# def select_best(s_phrase_lens, d_words, l_phrases, l_objs, iiter, l_record_scores, l_record_objs, best_other, b_do_dbs):
# 	min_score, max_score = sys.float_info.max, -sys.float_info.max
# 	num_objs = len(l_objs)
# 	l_scores = []
# 	for iobj in range(num_objs):
# 		if b_do_dbs:
# 			nd_bit_db = l_objs[iobj]
# 			d_phrase_sel_mats = best_other
# 		else:
# 			nd_bit_db = best_other
# 			d_phrase_sel_mats = l_objs[iobj]
# 		score = score_db_and_sel_mat(s_phrase_lens, d_words, l_phrases, nd_bit_db, d_phrase_sel_mats)
# 		l_scores.append(score)
# 		if score > max_score:
# 			max_score = score
# 		if score < min_score:
# 			min_score = score
#
# 	# print('avg score:', np.mean(l_scores)) # , 'list', l_scores)
# 	print('iiter', iiter, 'avg score:', np.mean(l_scores), 'max score:', np.max(l_scores)) # , 'list', l_scores)
# 	if l_record_scores == [] or max_score > l_record_scores[0]:
# 		l_record_scores.insert(0, max_score)
# 		l_record_objs.insert(0, l_objs[l_scores.index(max_score)])
# 	else:
# 		l_objs[l_scores.index(min_score)] = l_record_objs[0]
# 		l_scores[l_scores.index(min_score)] = l_record_scores[0]
# 	# mid_score = (max_score + min_score) / 2.0
# 	mid_score = l_scores[np.array(l_scores).argsort()[c_mid_score]]
# 	if max_score == min_score:
# 			range_scores = max_score
# 			l_obj_scores = np.ones(len(l_scores), dtype=np.float32)
# 	elif mid_score == max_score:
# 		range_scores = max_score - min_score
# 		l_obj_scores = np.array([(score - min_score) / range_scores for score in l_scores])
# 	else:
# 		range_scores = max_score - mid_score
# 		l_obj_scores = np.array([(score - mid_score) / range_scores for score in l_scores])
# 	l_obj_scores = np.where(l_obj_scores > 0.0, l_obj_scores, np.zeros_like(l_obj_scores))
# 	sel_prob = l_obj_scores/np.sum(l_obj_scores)
# 	l_sel_dbs = np.random.choice(num_objs, size=num_objs, p=sel_prob)
# 	l_objs[:] = [copy.deepcopy(l_objs[isel]) for isel in l_sel_dbs]

# def mutate_dbs(l_dbs, num_uniques):
# 	num_flip_muts = int(c_db_num_flip_muts * num_uniques)
#
# 	for idb, nd_bit_db in enumerate(l_dbs):
# 		if random.random() < c_db_rnd_asex:
# 			for imut in range(num_flip_muts):
# 				allele, target = random.randint(0, c_bitvec_size - 1), random.randint(0, num_uniques-1)
# 				nd_bit_db[target][allele] = 1 if (nd_bit_db[target][allele] == 0) else 0
# 		elif random.random() < c_db_rnd_sex:
# 			partner_db = copy.deepcopy(random.choice(l_dbs))  # not the numpy function
# 			for allele in range(c_bitvec_size):
# 				for iun in range(num_uniques):
# 					if random.random() < 0.5:
# 						nd_bit_db[iun,:] = partner_db[iun,:]
#
# def mutate_sel_mats(l_d_phrase_sel_mats, s_phrase_lens):
# 	for isel, d_phrase_sel_mats in enumerate(l_d_phrase_sel_mats):
# 		for ilen, phrase_len in enumerate(s_phrase_lens):
# 			num_input_bits = ((phrase_len - 1) * c_bitvec_size) + (c_num_replicate_missing * phrase_len)
# 			sel_mat = d_phrase_sel_mats[phrase_len]
# 			if random.random() < c_rnd_asex:
# 				for imut in range(c_num_incr_muts):
# 					allele = random.randint(0, c_bitvec_size-1)
# 					num_bits = len(sel_mat[allele][0])
# 					if sel_mat[allele][1] < num_bits-2:
# 						sel_mat[allele][1] += 1
# 				for imut in range(c_num_incr_muts):
# 					allele = random.randint(0, c_bitvec_size-1)
# 					if sel_mat[allele][1] > 1:
# 						sel_mat[allele][1] -= 1
# 				for icmut in range(c_num_change_muts):
# 					allele = random.randint(0, c_bitvec_size-1)
# 					bit_list = sel_mat[allele][0]
# 					if random.random() < c_change_mut_prob_change_len:
# 						if len(bit_list) < c_max_bits:
# 							bit_list.append(random.randint(0, num_input_bits - 1))
# 					elif random.random() < c_change_mut_prob_change_len:
# 						if len(bit_list) > c_min_bits:
# 							bit_list.pop(random.randrange(len(bit_list)))
# 							if sel_mat[allele][1] >= len(bit_list) - 1:
# 								sel_mat[allele][1] -= 1
# 					else:
# 						for ichange in range(c_change_mut_num_change):
# 							bit_list[random.randint(0, len(bit_list)-1)] = random.randint(0, num_input_bits - 1)
# 			elif random.random() < c_rnd_sex:
# 				partner_sel_mat = copy.deepcopy(random.choice(l_d_phrase_sel_mats)[phrase_len]) # not the numpy function
# 				for allele in range(c_bitvec_size):
# 					if random.random() < 0.5:
# 						sel_mat[allele] = list(partner_sel_mat[allele])

def main():
	# success_orders_freq = dict()
	freq_tbl, s_phrase_lens = load_order_freq_tbl(fnt)
	init_len = c_init_len # len(freq_tbl) / 2
	d_words = create_word_dict(freq_tbl, init_len)
	num_uniques = len(d_words)
	nd_bit_db = np.random.choice(a=[0, 1], size=(num_uniques, c_bitvec_size))
	l_change_db = [[[0.0 for ibit in xrange(c_bitvec_size)], 0.0] for _ in xrange(num_uniques)]
	# d_phrase_sel_mats, d_lens = dict(), dict()
	# for ilen, phrase_len in enumerate(s_phrase_lens):
	# 	num_input_bits = phrase_len * c_bitvec_size
	# 	sel_mat = []
	# 	for ibit in range(c_bitvec_size):
	# 		num_bits = random.randint(c_min_bits, c_max_bits)
	# 		l_sels = []
	# 		for isel in range(num_bits):
	# 			l_sels.append(random.randint(0, num_input_bits-1))
	# 		sel_mat.append([l_sels, random.randint(1, num_bits)])
	# 	d_phrase_sel_mats[phrase_len] = sel_mat
	# 	d_lens[phrase_len] = ilen
	d_lens = {phrase_len:ilen for ilen, phrase_len in enumerate(s_phrase_lens)}

	l_phrases = [[] for _ in s_phrase_lens]
	for iphrase, phrase in enumerate(freq_tbl):
		if iphrase >= init_len:
			break
		plen = len(phrase)
		l_phrases[d_lens[plen]].append(phrase)


	# l_d_phrase_sel_mats = [copy.deepcopy(d_phrase_sel_mats) for isel in range(c_num_sel_mats)]
	# l_record_sel_mat_scores = [-1.0]
	# l_record_sel_mats = [d_phrase_sel_mats]
	# l_record_db_scores = []
	# l_record_dbs = []

	for iiter in range(c_num_iters):
		score = score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, l_change_db)
		print('iiter', iiter, 'score:', score)  # , 'list', l_scores)
	# select_best(l_objs, iiter, l_record_scores, l_record_objs, best_other, b_do_dbs)
		# if iiter % 1 == 0:
		# 	# if iiter == 0:
		# 	select_best(s_phrase_lens, d_words, l_phrases, l_dbs, iiter,
		# 				l_record_db_scores, l_record_dbs, l_record_sel_mats[0], b_do_dbs=True)
		# 	mutate_dbs(l_dbs, num_uniques)
		# # else:
		# # 	select_best(s_phrase_lens, d_words, l_skip_phrases, l_d_phrase_sel_mats, iiter,
		# # 				l_record_sel_mat_scores, l_record_sel_mats, l_record_dbs[0], b_do_dbs=False)
		# # 	mutate_sel_mats(l_d_phrase_sel_mats, s_phrase_lens)

main()
print('done')
