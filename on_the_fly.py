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
import os
from os.path import expanduser
from shutil import copyfile
import itertools

import numpy as np

# fnt = 'orders_success.txt'
fnt = '~/tmp/adv_phrase_freq.txt'
fnt_dict = '~/tmp/adv_bin_dict.txt'

c_bitvec_size = 16
c_min_bits = 3
c_max_bits = 20
c_num_replicate_missing = 5
c_ham_winners_fraction = 32
c_num_iters = 10000 # 300000
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
c_init_len = 2000
c_move_rnd = 0.5
c_move_rnd_change = 0.02
c_min_frctn_change  = 0.001
c_max_frctn_change  = 0.01
c_b_init_db = True
c_save_init_db_every = 100
c_kmeans_divider_offset = 5
c_add_batch = 400
c_add_fix_iter = 20


def create_word_dict(phrase_list, max_process):
	d_els, l_presence, l_phrase_ids = dict(), [], []
	for iphrase, phrase in enumerate(phrase_list):
		if iphrase > max_process:
			break
		for iel, el, in enumerate(phrase):
			id = d_els.get(el, -1)
			if id == -1:
				d_els[el] = len(d_els)
				l_presence.append(1)
				l_phrase_ids.append([])
			else:
				l_presence[id] += 1
				# l_phrase_ids[id].append(iphrase)


	return d_els, l_presence, l_phrase_ids

def save_word_db(d_words, nd_bit_db):
	fn = expanduser(fnt_dict)

	if os.path.isfile(fn):
		copyfile(fn, fn + '.bak')
	fh = open(fn, 'wb')
	csvw = csv.writer(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	csvw.writerow(['Adv Dict', 'Version', '1'])
	csvw.writerow(['Num Els:', len(d_words)])
	for kword, virow in d_words.iteritems():
		csvw.writerow([kword, virow] + nd_bit_db[virow].tolist())

	fh.close()

def load_word_db():
	fn = expanduser(fnt_dict)
	try:
		with open(fn, 'rb') as o_fhr:
			csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, _, version_str = next(csvr)
			_, snum_els = next(csvr)
			version, num_els = int(version_str), int(snum_els)
			if version != 1:
				raise IOError
			d_words, s_word_bit_db, nd_bit_db = dict(), set(), np.zeros((num_els, c_bitvec_size), dtype=np.uint8)
			for irow, row in enumerate(csvr):
				word, iel, sbits = row[0], row[1], row[2:]
				d_words[word] = int(iel)
				bits = map(int, sbits)
				nd_bit_db[int(iel)] = np.array(bits, dtype=np.uint8)
				s_word_bit_db.add(tuple(bits))

	except IOError:
		raise ValueError('Cannot open or read ', fn)

	return d_words, nd_bit_db, s_word_bit_db

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
	# random.shuffle(freq_tbl)
	return freq_tbl, s_phrase_lens


def create_input_bits(nd_bit_db, d_words, phrase, l_b_known=[]):
	phrase_len = len(phrase)
	input_bits = np.zeros(phrase_len * c_bitvec_size, dtype=np.uint8)
	loc = 0
	for iword, word in enumerate(phrase):
		if l_b_known != [] and not l_b_known[iword]:
			input_bits[loc:loc + c_bitvec_size] = np.zeros(c_bitvec_size, dtype=np.uint8)
		else:
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


def score_hd_output_bits(nd_phrase_bits_db, qbits, mbits, iskip, iword, change_db, bscore=True):
	numrecs = nd_phrase_bits_db.shape[0]
	hd_divider = np.array(range(c_kmeans_divider_offset, score_hd_output_bits.num_ham_winners + c_kmeans_divider_offset),
						  np.float32)
	hd_divider_sum = np.sum(1. / hd_divider)

	def calc_score(outputs):
		odiffs = np.logical_and(np.not_equal(qbits, outputs), np.logical_not(mbits))
		nd_diffs = np.where(odiffs, np.ones_like(outputs), np.zeros_like(outputs))
		divider = np.array(range(1, nd_diffs.shape[0] + 1), np.float32)
		return np.sum(np.divide(np.sum(nd_diffs, axis=1), divider))

	# nd_diffs = np.absolute(np.subtract(qbits, nd_phrase_bits_db))
	nd_diffs = np.logical_and(np.not_equal(qbits, nd_phrase_bits_db), mbits)
	nd_diffs = np.where(nd_diffs, np.ones_like(nd_phrase_bits_db), np.zeros_like(nd_phrase_bits_db))
	hd = np.sum(nd_diffs, axis=1)
	hd_winners = np.argpartition(hd, score_hd_output_bits.num_ham_winners)[:score_hd_output_bits.num_ham_winners]
	hd_of_winners = hd[hd_winners]
	iwinners = np.argsort(hd_of_winners)
	hd_idx_sorted = hd_winners[iwinners]
	winner_outputs = nd_phrase_bits_db[hd_idx_sorted]
	avg_outputs = nd_phrase_bits_db[np.random.randint(numrecs, size=hd_idx_sorted.shape[0])]
	obits = winner_outputs[:, iskip*c_bitvec_size:(iskip+1)*c_bitvec_size]
	bad_obits = avg_outputs[:, iskip*c_bitvec_size:(iskip+1)*c_bitvec_size]
	# ibits = qbits[iskip*c_bitvec_size:(iskip+1)*c_bitvec_size].astype(float)
	# obits_goal = np.where(np.average(obits, axis=0) > 0.5, np.ones_like(ibits), np.zeros_like(ibits))
	obits_goal = np.sum(obits.transpose() / hd_divider, axis=1) / hd_divider_sum
	obits_keep_away = np.average(bad_obits, axis=0)
	new_obits_goal = ((obits_goal + (np.ones(c_bitvec_size) - obits_keep_away)) / 2.0).tolist()
	if change_db[iword][1] == 0.0:
		change_db[iword][0] = new_obits_goal
	else:
		change_db[iword][0] = ((np.array(change_db[iword][0]) * change_db[iword][1]) + new_obits_goal) / (change_db[iword][1] + 1.0)
	change_db[iword][1] += 1.0
	if not bscore:
		return
	close_score, avg_score = calc_score(winner_outputs), calc_score(avg_outputs)
	return avg_score / (close_score + 10.0)


score_hd_output_bits.num_ham_winners = 0

def build_bit_masks(s_phrase_lens):
	l_l_mbits = [] # mask bits
	for ilen, phrase_len in enumerate(s_phrase_lens):
		l_mbits = []
		for iskip in range(phrase_len):
			mbits = np.ones(phrase_len * c_bitvec_size, np.uint8)
			mbits[iskip*c_bitvec_size:(iskip+1)*c_bitvec_size] = np.zeros(c_bitvec_size, np.uint8)
			l_mbits.append(mbits)
		l_l_mbits.append(l_mbits)
	return l_l_mbits

def change_bit(nd_bit_db, s_word_bit_db, l_bits_now, l_bits_avg, iword):
	# if random.random() < score_and_change_db.move_rnd:
	bchanged = False
	ibit = random.randint(0, c_bitvec_size - 1)
	bit_now, bit_goal = l_bits_now[ibit], l_bits_avg[ibit]
	proposal = np.copy(nd_bit_db[iword])
	if bit_now == 0 and bit_goal > 0.5:
		if random.random() < ((bit_goal - 0.5) * 4):
			proposal[ibit] = 1
			bchanged = True
	elif bit_now == 1 and bit_goal < 0.5:
		if random.random() < ((0.5 - bit_goal) * 4):
			proposal[ibit] = 0
			bchanged = True
	if bchanged:
		tproposal = tuple(proposal.tolist())
		if tproposal not in s_word_bit_db:
			tremove = tuple(nd_bit_db[iword].tolist())
			nd_bit_db[iword, :] = proposal
			s_word_bit_db.remove(tremove)
			s_word_bit_db.add(tproposal)
			return 1
	return 0

def score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, s_word_bit_db):
	num_uniques = len(d_words)
	l_change_db = [[[0.0 for ibit in xrange(c_bitvec_size)], 0.0] for _ in xrange(num_uniques)]
	bitvec_size = nd_bit_db.shape[1]
	num_scored = 0
	num_hits = 0
	l_l_mbits = build_bit_masks(s_phrase_lens) # mask bits

	phrase_bits_db = [np.zeros((len(l_len_phrases), bitvec_size * list(s_phrase_lens)[ilen]), dtype=np.int)
					  for ilen, l_len_phrases in enumerate(l_phrases)]
	score = 0.0
	for ilen, phrase_len in enumerate(s_phrase_lens):
		num_scored += len(l_phrases[ilen]) * phrase_len
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
	score /= num_scored

	num_changed = 0
	for iunique, bits_data in enumerate(l_change_db):
		l_bits_avg, _ = bits_data
		l_bits_now = nd_bit_db[iunique]
		num_changed += change_bit(nd_bit_db, s_word_bit_db, l_bits_now, l_bits_avg, iunique)
		# if random.random() < score_and_change_db.move_rnd:
		# 	bchanged = False
		# 	ibit = random.randint(0, c_bitvec_size-1)
		# 	bit_now, bit_goal = l_bits_now[ibit], l_bits_avg[ibit]
		# 	proposal = np.copy(nd_bit_db[iunique])
		# 	if bit_now == 0 and bit_goal > 0.5:
		# 		if random.random() < (bit_goal - 0.5):
		# 			proposal[ibit] = 1
		# 			bchanged = True
		# 	elif bit_now == 1 and bit_goal < 0.5:
		# 		if random.random() < (0.5 - bit_goal):
		# 			proposal[ibit] = 0
		# 			bchanged = True
		# 	if bchanged:
		# 		tproposal = tuple(proposal.tolist())
		# 		if tproposal not in s_word_bit_db:
		# 			tremove = tuple(nd_bit_db[iunique].tolist())
		# 			nd_bit_db[iunique, :] = proposal
		# 			s_word_bit_db.remove(tremove)
		# 			s_word_bit_db.add(tproposal)
		# 			num_changed += 1
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

def build_phrase_bin_db(s_phrase_lens, l_phrases, nd_el_bin_db, d_words):
	phrase_bits_db = [np.zeros((len(l_len_phrases), c_bitvec_size * list(s_phrase_lens)[ilen]), dtype=np.int)
					  for ilen, l_len_phrases in enumerate(l_phrases)]
	score = 0.0
	for ilen, phrase_len in enumerate(s_phrase_lens):
		for iphrase, phrase in enumerate(l_phrases[ilen]):
			# nd_bits = np.zeros(c_bitvec_size, dtype=np.int)
			input_bits = create_input_bits(nd_el_bin_db, d_words, phrase)
			phrase_bits_db[ilen][iphrase, :] = input_bits

	return phrase_bits_db

def change_phrase_bin_db(phrase_bits_db, l_phrases, nd_el_bin_db, d_words, iword, l_word_phrase_ids):
	score = 0.0
	for ilen, iphrase in l_word_phrase_ids[iword]:
		phrase = l_phrases[ilen][iphrase]
		input_bits = create_input_bits(nd_el_bin_db, d_words, phrase)
		phrase_bits_db[ilen][iphrase, :] = input_bits


# ilen is the index number of the list of phrase grouped by phrase len (not the length of the phrase)
# iphrase is index in that list of phrases of that length
def add_new_words(	nd_bit_db, d_words, nd_phrase_bits_db, phrase, phrase_bits, s_word_bit_db,
					l_b_known, l_word_counts, l_word_phrase_ids, iphrase, ilen, l_change_db):
	divider = np.array(range(c_kmeans_divider_offset, score_hd_output_bits.num_ham_winners + c_kmeans_divider_offset),
					   np.float32)
	divider_sum = np.sum(1. / divider)
	phrase_len = len(l_b_known)
	mbits = np.ones(phrase_len * c_bitvec_size, np.uint8)
	for iword, bknown in enumerate(l_b_known):
		if not bknown:
			mbits[iword * c_bitvec_size:(iword + 1) * c_bitvec_size] = np.zeros(c_bitvec_size, np.uint8)

	nd_diffs = np.logical_and(np.not_equal(phrase_bits, nd_phrase_bits_db), mbits)
	nd_diffs = np.where(nd_diffs, np.ones_like(nd_phrase_bits_db), np.zeros_like(nd_phrase_bits_db))
	hd = np.sum(nd_diffs, axis=1)
	hd_winners = np.argpartition(hd, score_hd_output_bits.num_ham_winners)[:score_hd_output_bits.num_ham_winners]
	hd_of_winners = hd[hd_winners]
	iwinners = np.argsort(hd_of_winners)
	hd_idx_sorted = hd_winners[iwinners]
	winner_outputs = nd_phrase_bits_db[hd_idx_sorted]
	for iword, bknown in enumerate(l_b_known):
		if not bknown:
			obits = winner_outputs[:, iword*c_bitvec_size:(iword+1)*c_bitvec_size]
			new_vals = np.sum(obits.transpose() / divider, axis=1) / divider_sum
			# round them all and if the pattern is already there switch the closest to 0.5
			new_bits = np.round_(new_vals).astype(np.uint8)
			if tuple(new_bits) in s_word_bit_db:
				bfound = False
				while True:
					can_flip = np.argsort(np.square(new_vals - 0.5))
					for num_flip in range(1, c_bitvec_size):
						try_flip = can_flip[:num_flip]
						l = [list(itertools.combinations(try_flip, r)) for r in range(num_flip+1)]
						lp = [item for sublist in l for item in sublist]
						for p in lp:
							pbits = list(new_bits)
							for itf in try_flip:
								pbits[itf] = 1 if itf in p else 0
							if tuple(pbits) not in s_word_bit_db:
								new_bits = pbits
								bfound = True
								break
						if bfound:
							break
					if bfound:
						break

			s_word_bit_db.add(tuple(new_bits))
			nd_bit_db = np.concatenate((nd_bit_db, np.expand_dims(new_bits, axis=0)), axis=0)
			d_words[phrase[iword]] = len(d_words)
			l_change_db += [[[0.0 for ibit in xrange(c_bitvec_size)], 0.0]]
			l_word_phrase_ids.append([(ilen, iphrase)])
			l_word_counts.append(1)
			pass
		else: # is known
			id = d_words[phrase[iword]]
			l_word_counts[id] += 1
			l_word_phrase_ids[id].append((ilen, iphrase))
	return nd_bit_db


def keep_going(	freq_tbl, d_words, nd_el_bin_db, s_word_bit_db, s_phrase_lens, l_phrases,
				l_word_counts, l_word_phrase_ids, add_start, num_add_batch):
	phrase_bin_db = build_phrase_bin_db(s_phrase_lens, l_phrases, nd_el_bin_db, d_words)
	l_change_db = [[[0.0 for _ in xrange(c_bitvec_size)], 0.0] for _ in l_word_counts]
	num_changed = 0
	for ifreq, phrase in enumerate(freq_tbl[add_start:add_start+num_add_batch]):
		for iel, el, in enumerate(phrase):
			phrase_len = len(phrase)
			if phrase_len not in s_phrase_lens:
				raise ValueError('New phrase len. Not coded this possibility yet')
			ilen = list(s_phrase_lens).index(phrase_len)
			iphrase = len(l_phrases[ilen])
			l_phrases[ilen].append(phrase)
			l_b_known = [True for _ in phrase]
			for iword, word in enumerate(phrase):
				id = d_words.get(word, -1)
				if id == -1:
					l_b_known[iword] = False
			if not all(l_b_known):
				phrase_bits = create_input_bits(nd_el_bin_db, d_words, phrase, l_b_known=l_b_known)
				nd_el_bin_db = add_new_words(nd_el_bin_db, d_words, phrase_bin_db[ilen], phrase, phrase_bits,
											 s_word_bit_db, l_b_known, l_word_counts, l_word_phrase_ids,
											 iphrase, ilen, l_change_db)
				input_bits = create_input_bits(nd_el_bin_db, d_words, phrase)
				phrase_bin_db[ilen] = np.concatenate((phrase_bin_db[ilen], np.expand_dims(input_bits, axis=0)), axis=0)
			else:
				l_l_mbits = build_bit_masks(s_phrase_lens)  # mask bits
				input_bits = create_input_bits(nd_el_bin_db, d_words, phrase)
				phrase_bin_db[ilen] = np.concatenate((phrase_bin_db[ilen], np.expand_dims(input_bits, axis=0)), axis=0)
				for iskip in range(phrase_len):
					iword = d_words[phrase[iskip]]
					score_hd_output_bits(	phrase_bin_db[ilen], phrase_bin_db[ilen][-1],
											l_l_mbits[ilen][iskip], iskip, iword,
											l_change_db, bscore=False)
					(l_bits_avg, num_hits), word_count = l_change_db[iword], l_word_counts[iword]

					if num_hits * 2 > word_count:
						bchanged = change_bit(nd_el_bin_db, s_word_bit_db, nd_el_bin_db[iword], l_bits_avg, iword)
						if bchanged == 1:
							num_changed += 1
							change_phrase_bin_db(phrase_bin_db, l_phrases, nd_el_bin_db, d_words, iword, l_word_phrase_ids)
						l_change_db[iword] = [[0.0 for ibit in xrange(c_bitvec_size)], 0.0]

	return nd_el_bin_db


def main():
	# success_orders_freq = dict()
	freq_tbl, s_phrase_lens = load_order_freq_tbl(fnt)
	init_len = c_init_len # len(freq_tbl) / 2
	d_words, l_word_counts, l_word_phrase_ids = create_word_dict(freq_tbl, init_len)
	num_ham_winners = len(d_words) / c_ham_winners_fraction
	score_hd_output_bits.num_ham_winners= num_ham_winners
	num_uniques = len(d_words)
	nd_bit_db = np.zeros((num_uniques, c_bitvec_size), dtype=np.uint8)
	s_word_bit_db = set()
	for iunique in range(num_uniques):
		while True:
			proposal = np.random.choice(a=[0, 1], size=(c_bitvec_size))
			tproposal = tuple(proposal.tolist())
			if tproposal in s_word_bit_db:
				continue
			nd_bit_db[iunique, :] = proposal
			s_word_bit_db.add(tproposal)
			break
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

	for ilen, phrases in enumerate(l_phrases):
		for iphrase, phrase in enumerate(phrases):
			for iel, el in enumerate(phrase):
				id = d_words[el]
				l_word_phrase_ids[id].append((ilen, iphrase))

	if c_b_init_db:
		for iiter in range(c_num_iters):
			score = score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, s_word_bit_db)
			print('iiter', iiter, 'score:', score)  # , 'list', l_scores)
			if iiter % c_save_init_db_every == 0:
				save_word_db(d_words, nd_bit_db)
		return
	else:
		d_words, nd_bit_db, s_word_bit_db = load_word_db()

	add_start = c_init_len
	while (add_start < len(freq_tbl)):
		nd_bit_db = keep_going(	freq_tbl, d_words, nd_bit_db, s_word_bit_db, s_phrase_lens,
								l_phrases, l_word_counts, l_word_phrase_ids, add_start, c_add_batch)
		print('Added', c_add_batch, 'phrases after', add_start)
		add_start += c_add_batch
		for iiter in range(c_add_fix_iter):
			score = score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, s_word_bit_db)
			print('iiter', iiter, 'score:', score)  # , 'list', l_scores)



main()
print('done')

