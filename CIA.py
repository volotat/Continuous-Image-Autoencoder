from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense
from keras.utils import plot_model
from keras import layers
from PIL import Image
import keras
import numpy as np
import math
import os.path
import sys
import winsound
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-f", "--dataset", dest="dataset", type="string",
					help="Filename of image dataset")
parser.add_option("-e", "--examples", dest="examples", type="int", 
					help="Number of examples in dataset")
parser.add_option("--no-train", dest="train", action="store_false", default=True,
					help="If only you want is get some results out of pretrained model")

(options, args) = parser.parse_args()

if (options.dataset == None or options.examples == None):
	print ('You should provide path to dataset file and number of examples in it. Example: "-f cars.png -e 20"')
	sys.exit()
	


SIGNAL_WHEN_COMPLETE = True

#INPUT/OUTPUT DATA PARAMETERS
TRAIN_DATASET = options.dataset #Image set in one image. Should have 10 images in row and as many rows as you want.
EXAMPLES = options.examples #Number of examples in dataset
TRAIN = options.train #Set False if only you want is get some results out of pretrained model
GENERATED_POWER = 2 #(from 1 to inf) if number is bigger, generated results will be closer to original data
size = 200 #Size of single image in input image set
half_size = math.floor(size / 2)

out_size = 64 #Size of single image in output image set
out_half_size = math.floor(out_size / 2)


#TRAIN INFO
EPOCHS = 8  #It's better to set epochs number slightly higher if input data not mixed. For me 8-10 for none-mixed and 4 for mixed data works best.
ADD_LAYER = True #If set True new layer will be added to the model before each training step
TRAIN_OLD = True #If set False only new layers will be trained
MIXED = False #Creates mix of inputs. No precise rule when you should use it, but it useful for face generation as example
MIXED_PURE_MULT = 1 #if it's need to provide dominance of original data in the mix


class C_GROW:
	TOP = 0
	MIDDLE = 1
	BOTTOM = 2 
GROW = C_GROW.TOP
GROW_STEPS = 5 #it's not necessary to set this parameter too high because you can easily run script multiple times if you not satisfied with results


#MODEL INFO
BATCH_SIZE = 2 ** 14
enc_name = 'encoder.h5'
dec_name = 'decoder.h5'

LATENT_SPACE = 3
INPUT_SPACE = 2

position_input = Input(shape = (INPUT_SPACE,), name = 'input_position')
img_ident_input = Input(shape = (EXAMPLES,), name = 'one_hot_img_ident')
inputs = [position_input, img_ident_input]


#GLOBAL VARIABLES
encoder 	= None
decoder		= None
redecoder 	= None
lat_arr 	= None
im_arr 		= None
grid		= None
noise		= None
weights 	= None
init_dec	= None

loss 		= np.inf
grow_step 	= 0
first_step  = False
_EPOCHS 	= EPOCHS

def set_up_model():
	global encoder
	global decoder
	global redecoder
	global ADD_LAYER
	global first_step
	global weights
	
	if (not os.path.isfile(enc_name)):
		
		encoded_posit = Dense(64, activation='tanh', name = 'enc_pos')(position_input) 
		#We want to increase dimensionality of input position in order to get more sharpest results
		
		encoded_ident = Dense(LATENT_SPACE, activation='tanh', name = 'enc_lat')(img_ident_input)  
		#This layer generate latent space, you could replace it with LATENT_SPACE-dimensional input after training and use it as input
		
		encoded = keras.layers.concatenate([encoded_posit, encoded_ident])
		encoder = Model(inputs, encoded)
		encoder.compile(optimizer='adam', loss='mse')
		
		dec_input = Input(shape = (64 + LATENT_SPACE,))
		
		decoded = Dense(9, activation='tanh', name = 'dec_in_1')(dec_input) 
		if (GROW == C_GROW.MIDDLE):
			decoded = Dense(9, activation='tanh', name = 'dec_ot_1')(decoded) 
			
			
			
		decoded = Dense(3, activation='tanh', name = 'dec_out')(decoded) 
		decoder = Model(dec_input, decoded)
		decoder.compile(optimizer='adam', loss='mse')
		
		
		redecoder = Model(inputs, decoder(encoder(inputs)))
		redecoder.compile(optimizer='adam', loss='mse')
		
		plot_model(encoder, to_file='encoder.png', show_shapes=True)
		plot_model(decoder, to_file='decoder.png', show_shapes=True)
		
		#No more layers at first step
		first_step = True
	else:
		encoder = load_model(enc_name)
		decoder = load_model(dec_name)	
			
		redecoder = Model(encoder.inputs, decoder(encoder(encoder.inputs)))
		redecoder.compile(optimizer='adam', loss='mse')
		
	weights = redecoder.get_weights()
		
def add_layers():
	global encoder
	global decoder
	global redecoder
	global _EPOCHS
	global init_dec
	global size
	global half_size
	
	init_dec = decoder.get_config()
	
	layers_count = len(decoder.layers)
	
	if (GROW == C_GROW.TOP or GROW == C_GROW.BOTTOM):
		power = layers_count - 1
	else:
		power = layers_count // 2
		
	enc_neurons = 64 + LATENT_SPACE 
	dec_neurons = 27 * (power - 1)
	_EPOCHS = (EPOCHS // 2) * (power + 1)
	
	layers_count = len(decoder.layers)
	dec_input = Input(shape = (enc_neurons,))
	decoded = dec_input
	
	
	#Grow in top
	if (GROW == C_GROW.TOP):
		decoded = Dense(dec_neurons, activation='tanh', name = 'dec_z_'+str(power))(decoded)
		for i in range(1, layers_count):
			config = decoder.layers[i].get_config()
			decoded = Dense.from_config(config)(decoded)
			decoded.trainable = TRAIN_OLD
			
	#Grow in middle
	if (GROW == C_GROW.MIDDLE):
		for i in range(1, layers_count):
			config = decoder.layers[i].get_config()
			decoded = Dense.from_config(config)(decoded)
			decoded.trainable = TRAIN_OLD
			if i == (layers_count) // 2 - 1:
				decoded = Dense(dec_neurons, activation='tanh', name = 'dec_z_in_'+str(power))(decoded)
				decoded = Dense(dec_neurons, activation='tanh', name = 'dec_z_ot_'+str(power))(decoded)
				
	#Grow in bottom
	if (GROW == C_GROW.BOTTOM):
		if layers_count == 3:
			decoded = Dense(dec_neurons, activation='tanh', name = 'dec_z_ot_'+str(power))(decoded)
			
		for i in range(1, layers_count):
			config = decoder.layers[i].get_config()
			decoded = Dense.from_config(config)(decoded)
			decoded.trainable = TRAIN_OLD
			if i == layers_count - 3:
				decoded = Dense(dec_neurons, activation='tanh', name = 'dec_z_ot_'+str(power))(decoded)
	
	decoder = Model(dec_input, decoded)
	
	
	redecoder = Model(inputs, decoder(encoder(inputs)))
	redecoder.compile(optimizer='adam', loss='mse')
	
	plot_model(encoder, to_file='encoder.png', show_shapes=True)
	plot_model(decoder, to_file='decoder.png', show_shapes=True)
	
def prepare_data():
	global lat_arr
	global im_arr
	global grid
	global noise

	lat_arr = np.zeros((EXAMPLES, EXAMPLES))
	pnt = np.arange(EXAMPLES)
	lat_arr[pnt, pnt] = 1 #one hot matrix for representing pictures			
		
	im_arr = np.zeros((EXAMPLES, size * size, 3))
	im_set = Image.open(TRAIN_DATASET).convert('RGB')
	wh = (im_set.size[0] / size, im_set.size[1] / size)

	for i in range(EXAMPLES):
		w = i % wh[0] * size
		h = math.floor (i / wh[0]) * size
		im = im_set.crop((w, h, w + size, h + size))
		im_arr[i] = np.array(im).reshape(size * size, 3) / 255.
		
		
	if (not MIXED):
		ident_space = lat_arr
		image_space = im_arr
	else:
		ident_space = np.zeros((EXAMPLES ** 2 + EXAMPLES * (MIXED_PURE_MULT - 1), EXAMPLES))	
		image_space  = np.zeros((EXAMPLES ** 2 + EXAMPLES * (MIXED_PURE_MULT - 1), size * size, 3))	
		
		ind = 0
		ind_x, ind_y = 0, 0

		for i in range(EXAMPLES):
			for j in range(EXAMPLES):
				if i == j:
					for k in range(MIXED_PURE_MULT):
						ident_space[ind] = (lat_arr[i] + lat_arr[j]) / 2.
						image_space [ind] = im_arr[i]
						ind += 1
						ind_x += 1
				else:
					#We need to have different results for same latent space in order to reproduce mix beetween any two images in a right way
					ident_space[ind] = (lat_arr[i] + lat_arr[j]) / 2.
					image_space[ind]  = im_arr[i]
					ind += 1
					ind_y += 1
						
		print ('Total number of exaples: ', ind, ind_x, ind_y)


	shape_0 = ident_space.shape[0]
	ident_space = ident_space.reshape(shape_0, 1, EXAMPLES)
	print ('ident_space shape:', ident_space.shape)
	im_arr = image_space.reshape(shape_0 * size * size, 3)
	im_arr = (im_arr-0.5) # [-0.5, 0.5]
	print ('im_arr shape:', im_arr.shape)	


	noise = ident_space
	noise = np.repeat(noise, size * size, axis=1).reshape((shape_0) * size ** 2, EXAMPLES)
	print ('noise shape:', noise.shape)	

	X,Y = np.mgrid[-half_size:half_size,-half_size:half_size] + 0.5
	grid = np.vstack((X.flatten(), Y.flatten())).T / half_size
	grid = grid.reshape(1, (half_size * 2) ** 2, 2)
	grid = np.repeat(grid, shape_0, axis=0).reshape((shape_0) * (half_size * 2) ** 2, 2)
	print ('grid shape:', grid.shape)	

def train_model():
	global lat_arr
	global im_arr
	global grid
	global noise
	global loss
	global weights
	global init_dec
	global decoder
	global redecoder
	
	
	is_improved = False
	for i in range(_EPOCHS):
		data = redecoder.fit([grid, noise], im_arr, batch_size = BATCH_SIZE, epochs=1, shuffle = True)
		n_loss = data.history['loss'][0]
		if n_loss<loss:
			weights = redecoder.get_weights()
			loss = n_loss
			is_improved = True
			
		print ('STEP: ',grow_step,'/',GROW_STEPS,'  //  EPOCHS: ',(i+1),'/',_EPOCHS,'  //  MIN_LOSS: ',loss)
		
		
	if is_improved:
		redecoder.set_weights(weights)
		encoder.save(enc_name)
		decoder.save(dec_name)
	else:
		#return model to state before last layer was added if loss did not become smaller
		decoder = Model.from_config(init_dec)
		
		redecoder = Model(inputs, decoder(encoder(inputs)))
		redecoder.compile(optimizer='adam', loss='mse')
		
		redecoder.set_weights(weights)
	
def show_results(out_img = 'out.bmp'):
	X,Y = np.mgrid[-out_half_size:out_half_size,-out_half_size:out_half_size] + 0.5
	grid = np.vstack((X.flatten(), Y.flatten())).T / out_half_size


	noise = np.random.uniform(0,1,(100, 1, EXAMPLES))
	for i in range(100):
		noise[i] = np.power(noise[i], GENERATED_POWER)
		noise[i] = noise[i] / np.sum(noise[i]) 
		
		
	_EXAMPLES = EXAMPLES #30
	noise[:_EXAMPLES] = lat_arr[:_EXAMPLES].reshape(_EXAMPLES, 1, EXAMPLES)


	a = np.arange(EXAMPLES)
	np.random.shuffle(a)

	for k in range(0, 5):
		im_x = lat_arr[a[k]]
		im_y = lat_arr[a[k+5]]
		
		for i in range(10):
			y = i / 9.
			x = 1. - y
			ind = 10+10*k + i
			noise[ind] = im_x*x + im_y*y
			noise[ind] = noise[ind] / np.sum(noise[ind]) 
		

	im_out = Image.new("RGB", (out_size * 10, out_size * 10))
	for i in range(100):
		c_noise = noise[i]
		c_noise = np.repeat(c_noise, grid.shape[0], axis=0)

		predicted = redecoder.predict([grid, c_noise]) + 0.5
		predicted = np.clip(predicted, 0, 1)
		predicted = (predicted * 255).astype(np.uint8).reshape(out_size, out_size, 3) 
		im = Image.fromarray(predicted)
		im_out.paste(im, (i % 10 * out_size, math.floor(i / 10) * out_size))
		
	im_out.save(out_img)

	if SIGNAL_WHEN_COMPLETE:
		winsound.Beep(600,800)
		
	

set_up_model()
prepare_data()

if TRAIN:
	loss = redecoder.evaluate([grid, noise], im_arr, batch_size = BATCH_SIZE)
	print ('START LOSS: ', loss)

	while (grow_step < GROW_STEPS):
		grow_step += 1
		
		print('\n')
			
		if (ADD_LAYER and not first_step):
			add_layers()
		first_step = False
		
		train_model()		
		show_results()
else:
	show_results()