import tensorflow as tf
import numpy as np
import sys
import getopt
from conllfeatures import *
from Scorer import F1_scores
from os import listdir
from os.path import isfile, join

# Config Variables
PHIA_FEATURE_LEN = 200
PHIP_FEATURE_LEN = 200
WA_WIDTH = 128
WP_WIDTH = 128
FL_PENALTY = 0.5
FN_PENALTY = 1.2
WL_PENALTY = 1
LEARNING_RATE = 0.01
W2V_MIN_COUNT = 1
W2V_SIZE = 200
W2V_WINDOW = 5
ITERATION_COUNT = 1
TRAIN_DIR = "./Data/Train/"
TEST_DIR = "./Data/Test/"
CKPT_PATH = "./Checkpoints/local.ckpt"
RESTORE = False
SAVE = False
HIDDEN_SIZE = 200

opts, args = getopt.getopt(sys.argv[1:],"n:l:d:rs:",[])
for opt, arg in opts:
	if opt == '-n':
		ITERATION_COUNT = int(arg)
	elif opt == '-l':
		LEARNING_RATE = float(arg)
	elif opt == '-d':
		TRAIN_DIR = arg + "/Train"
		TEST_DIR = arg + "/Test"
	elif opt == '-r':
		RESTORE = True
	elif opt == '-s':
		SAVE = True
		CKPT_PATH = arg

train_wordfiles = filter(lambda filename:  filename.endswith('wordsList.txt') , listdir(TRAIN_DIR))
test_wordfiles = filter(lambda filename:  filename.endswith('wordsList.txt') , listdir(TEST_DIR))
NUM_FILES = len(train_wordfiles)

# Build Model for Local Mention Ranking
# Inputs/Placeholders (assuming we train one mention at a time)
# Here phia/p are the feature embeddings while Y is the best antecedent (or should we take cluster instead? - depends on output)
Phia_x = tf.placeholder(tf.float32, [1, PHIA_FEATURE_LEN])
Phip_x = tf.placeholder(tf.float32, [None, PHIP_FEATURE_LEN])

# Y_antecedent array has True where it belongs to the same cluster and False otherwise
Y_antecedent = tf.placeholder(tf.float32, [None, 1])

tr_size = tf.shape(Phip_x)[0]

# LSTM stuff
state_array = tf.Variable(tf.zeros([0, HIDDEN_SIZE]))
initial_state = tf.Variable(tf.zeros([1, HIDDEN_SIZE]))
cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=False)

# Variables/Parameters
W_a = tf.Variable(tf.random_uniform([PHIA_FEATURE_LEN, WA_WIDTH]), name="W_a")
b_a = tf.Variable(tf.random_uniform([1, WA_WIDTH]), name="b_a") 
W_p = tf.Variable(tf.random_uniform([PHIP_FEATURE_LEN, WP_WIDTH]), name="W_p")
b_p = tf.Variable(tf.random_uniform([1, WP_WIDTH]), name="b_p")

W_c = tf.Variable(tf.random_uniform([PHIA_FEATURE_LEN, HIDDEN_SIZE]), name="W_c")
b_c = tf.Variable(tf.random_uniform([1, HIDDEN_SIZE]), name="b_c")
W_s = tf.Variable(tf.random_uniform([PHIA_FEATURE_LEN + HIDDEN_SIZE, WA_WIDTH]), name="W_s")
b_s = tf.Variable(tf.random_uniform([1, WA_WIDTH]), name="b_s")

u = tf.Variable(tf.random_uniform([WA_WIDTH + WP_WIDTH, 1]), name="u")
v = tf.Variable(tf.random_uniform([WA_WIDTH, 1]), name="v")
q = tf.Variable(tf.random_uniform([WA_WIDTH, 1]), name="q")

b_u = tf.Variable(tf.random_uniform([1]), name="b_u")
b_v = tf.Variable(tf.random_uniform([1]), name="b_v")

# Get inner linear function Wa(x)+ba and Wp(x)+bp
l_a = tf.add(tf.matmul(Phia_x, W_a),b_a)
l_a_tiled = tf.tile(l_a, [tr_size, 1])

l_p = tf.add(tf.matmul(Phip_x, W_p), tf.tile(b_p, [tr_size, 1]))
l_p_concat = tf.concat(1, [l_a_tiled, l_p])

# LSTM part
h_c = tf.nn.tanh(tf.add(tf.matmul(Phia_x, W_c), b_c))
NA = tf.tanh(tf.add(tf.matmul(tf.concat(1, [Phia_x, tf.reshape(tf.reduce_sum(state_array, 0), [1, 200])]), W_s), b_s))

g_x_ana = tf.transpose(tf.matmul(h_c, tf.transpose(state_array)))
g_x_nonana = tf.matmul(NA, q)
g_x = tf.concat(0, [tf.fill([1,1], g_x_nonana[0][0]) ,g_x_ana])

# Fill best antecedent using max and all
f_x_ana = tf.add(tf.matmul(tf.nn.tanh(l_p_concat), u), tf.fill([tr_size, 1], b_u[0]))
f_x_nonana = tf.add(tf.matmul(tf.nn.tanh(l_a), v), b_v)

# Get argmax and max of ana and nonana f_x concatenated
f_x = tf.add(tf.concat(0, [tf.fill([1,1], f_x_nonana[0][0]) ,f_x_ana]), g_x)
best_ant = tf.argmax(f_x, 0)
f_x_best = tf.reduce_max(f_x, 0)

# Assign value to Y_antecedent somehow
f_x_reduced = tf.mul(f_x, Y_antecedent)
f_y_latent = tf.reduce_max(f_x_reduced,0)
y_latent = tf.argmax(f_x_reduced,0)

loss_multiplier = tf.select(tf.equal(y_latent,tf.constant(0, dtype='int64'))[0], tf.constant(FL_PENALTY, dtype='float32'), tf.select(tf.equal(best_ant, tf.constant(0, dtype='int64'))[0],tf.constant(FN_PENALTY, dtype='float32'),tf.constant(WL_PENALTY, dtype='float32')))
loss_factor = tf.select(tf.equal(y_latent,best_ant)[0], tf.constant(0, dtype='float32'), loss_multiplier) 

loss = tf.mul(tf.add(tf.constant(1.0), tf.sub(f_x_best, f_y_latent)), loss_factor)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

saver = tf.train.Saver()

# Train model
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	if (RESTORE == True):
		saver.restore(sess, CKPT_PATH)
	for iteration_count in range(ITERATION_COUNT):
		for train_doc in train_wordfiles:

			wordFile = TRAIN_DIR + train_doc
			mentionFile = wordFile.replace("wordsList", "mentionsList")
			try:
				cluster_data = getClustersArrayForMentions(mentionFile)
				mentionFeats = getMentionFeats2(mentionFile,wordFile,W2V_MIN_COUNT,W2V_SIZE,W2V_WINDOW)
			except:
				print "Error on",train_doc
				continue

			TRAINING_SIZE = len(cluster_data)

			for i in range(TRAINING_SIZE):

				latent_antecedents = np.logical_not(cluster_data[:i] - cluster_data[i]).astype(np.int)
				latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([i+1,1])

				ant = np.array(sess.run(best_ant, feed_dict={Phia_x: mentionFeats[i].reshape(1,PHIA_FEATURE_LEN) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents}))
				
				sess.run(train_op, feed_dict={Phia_x: mentionFeats[i].reshape(1,PHIA_FEATURE_LEN), Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE), Y_antecedent: latent_antecedents})
				
				# Psuedocode Here MUAHAHAHAHA
				LSTM_inp = tf.identity(tf.Variable(mentionFeats[i].reshape(1,PHIA_FEATURE_LEN), dtype='float32'))
				if (ant == 0):
					_, state = cell(LSTM_inp, initial_state)
					state_array = tf.concat(0, [state_array, state])
				else:
					_, state = cell(LSTM_inp, state_array[ant-1])
					state_array = tf.concat(0, [state_array, state])

		eval_prec_b3 = 0
		eval_rec_b3 = 0
		eval_f1_b3 = 0
		eval_prec_muc = 0
		eval_rec_muc = 0
		eval_f1_muc = 0
		num_files = 0

		for test_doc in test_wordfiles:

			wordFile = TEST_DIR + test_doc
			mentionFile = wordFile.replace("wordsList", "mentionsList")
				
			cluster_data = getClustersArrayForMentions(mentionFile)
			mentionFeats = getMentionFeats2(mentionFile,wordFile,W2V_MIN_COUNT,W2V_SIZE,W2V_WINDOW)

			TRAINING_SIZE = len(cluster_data)

			cluster_pred = np.zeros(TRAINING_SIZE)
			score = 0

			for i in range(TRAINING_SIZE):

				latent_antecedents = np.logical_not(cluster_data[:i] - cluster_data[i]).astype(np.int)
				latent_antecedents = np.append(np.array([not latent_antecedents.any()]).astype(np.int), latent_antecedents).reshape([i+1,1])

				cluster_pred[i] = np.array(sess.run(best_ant, feed_dict={Phia_x: mentionFeats[i].reshape(1,PHIA_FEATURE_LEN) ,Phip_x: getPairFeats(i, mentionFeats, W2V_SIZE) ,Y_antecedent: latent_antecedents}))

				if (cluster_pred[i] == 0):
					score = score + 1
					for j in range(i):
						if (cluster_data[j] == cluster_data[i]):
							score = score - 1
							break
				elif (cluster_data[cluster_pred[i]-1] == cluster_data[i]):
					score = score + 1

			# print wordFile
			(f1_b3, rec_b3, prec_b3, f1_muc, rec_muc, prec_muc, f1_ceaf, rec_ceaf, prec_ceaf) = F1_scores(cluster_data, cluster_pred)
			# print score, rec, prec, (score*100.0)/TRAINING_SIZE
			eval_rec_b3 += rec_b3
			eval_prec_b3 += prec_b3
			eval_f1_b3 += f1_b3
			eval_rec_muc += rec_muc
			eval_prec_muc += prec_muc
			eval_f1_muc += f1_muc
			eval_f1_ceaf += f1_ceaf
			eval_rec_ceaf += rec_ceaf
			eval_prec_ceaf += prec_ceaf
			num_files += 1


		print "Macro-ave B3 recall :", eval_rec_b3/num_files
		print "Macro-ave B3 precision :", eval_prec_b3/num_files
		print "Macro-ave B3 F1 :", eval_f1_b3/num_files

		print "Macro-ave MUC recall :", eval_rec_muc/num_files
		print "Macro-ave MUC precision :", eval_prec_muc/num_files
		print "Macro-ave MUC F1 :", eval_f1_muc/num_files

		print "Macro-ave CEAF recall :", eval_rec_ceaf/num_files
		print "Macro-ave CEAF precision :", eval_prec_ceaf/num_files
		print "Macro-ave CEAF F1 :", eval_f1_ceaf/num_files
	if (SAVE == True):
		saver.save(sess, CKPT_PATH)
	print "OVER"