import numpy

from keras.layers import Input, Embedding, Flatten, dot
from keras.models import Model

def matrix_data_generator(X, batch_size=128):
	rows, cols = numpy.where(~numpy.isnan(X))
	n = rows.shape[0]

	while True:
		idxs = numpy.random.choice(n, size=batch_size)
		
		d = {
			'row_input': rows[idxs],
			'col_input': cols[idxs]
		}

		yield d, X[rows[idxs], cols[idxs]]


class MatrixFactorization():
	def __init__(self, k, batch_size=128, n_epochs=50, optimizer='adam', 
		loss='mse', metrics=['mse'], verbose=True):
		self.k = k
		self.model = None
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics
		self.verbose = verbose

	@property
	def row_embedding(self):
		for layer in self.model.layers:
			if layer.name == 'row_embedding':
				return layer.get_weights()[0]

		raise ValueError("No layer in model named 'row_embedding'.")

	@property
	def col_embedding(self):
		for layer in self.model.layers:
			if layer.name == 'col_embedding':
				return layer.get_weights()[0].T

		raise ValueError("No layer in model named 'col_embedding'.")

	def _build_model(self, n_rows, n_cols):
		row_input = Input(shape=(1,), name="row_input")
		col_input = Input(shape=(1,), name="col_input")

		row_embedding = Embedding(n_rows, self.k, input_length=1,
			name="row_embedding")
		col_embedding = Embedding(n_cols, self.k, input_length=1,
			name="col_embedding")

		row = Flatten()(row_embedding(row_input))
		col = Flatten()(col_embedding(col_input))

		layers = [row, col]
		inputs = (row_input, col_input)

		y_hat = dot([row, col], axes=1, normalize=False, name="y_hat")
		model = Model(inputs=inputs, output=y_hat)
		model.compile(optimizer=self.optimizer, loss=self.loss, 
			metrics=self.metrics)
		return model

	def fit(self, X):
		n_rows, n_cols = X.shape
		X_generator = matrix_data_generator(X, self.batch_size)

		n_elems = (~numpy.isnan(X)).sum()
		epoch_size =  n_elems // self.batch_size + 1

		self.model = self._build_model(n_rows, n_cols)
		history = self.model.fit_generator(X_generator, epoch_size, self.n_epochs,
			workers=1, pickle_safe=True, verbose=self.verbose)

		return history


n_rows, n_cols, k = 100, 100, 10
n_missing = 1000

real_row_embeddings = numpy.random.randn(n_rows, k)
real_col_embeddings = numpy.random.randn(k, n_cols)

X = real_row_embeddings.dot(real_col_embeddings)

missing_rows = numpy.random.choice(n_rows, size=n_missing)
missing_cols = numpy.random.choice(n_cols, size=n_missing)

X[missing_rows, missing_cols] = numpy.nan

model = MatrixFactorization(10)
model.fit(X)

print(((model.row_embedding - real_row_embeddings) ** 2).mean())
print(((model.col_embedding - real_col_embeddings) ** 2).mean())