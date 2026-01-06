# lstm_numpy.py
#
# Manual LSTM implementation using NumPy.
# Contains an LSTMCell (single timestep) and LSTMNetwork (sequence model).
# This code avoids external deep learning libraries and uses SGD for updates.

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    """
    Single LSTM cell that processes one timestep.
    Stores intermediate values needed for BPTT.
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight initialization (standard uniform)
        limit = 1.0 / np.sqrt(hidden_size)

        # Gates: input, forget, output, candidate
        self.W_i = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.W_f = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.W_o = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.W_c = np.random.uniform(-limit, limit, (hidden_size, input_size))

        self.U_i = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_f = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_o = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_c = np.random.uniform(-limit, limit, (hidden_size, hidden_size))

        self.b_i = np.zeros((hidden_size,))
        self.b_f = np.zeros((hidden_size,))
        self.b_o = np.zeros((hidden_size,))
        self.b_c = np.zeros((hidden_size,))

    def forward(self, x_t, h_prev, c_prev):
        """
        Forward pass for a single timestep.
        x_t: (input_size,)
        h_prev: (hidden_size,)
        c_prev: (hidden_size,)
        Returns: h_t, c_t
        """

        # Gate activations
        self.x_t = x_t
        self.h_prev = h_prev
        self.c_prev = c_prev

        self.i_t = sigmoid(self.W_i @ x_t + self.U_i @ h_prev + self.b_i)
        self.f_t = sigmoid(self.W_f @ x_t + self.U_f @ h_prev + self.b_f)
        self.o_t = sigmoid(self.W_o @ x_t + self.U_o @ h_prev + self.b_o)
        self.c_hat_t = np.tanh(self.W_c @ x_t + self.U_c @ h_prev + self.b_c)

        # Cell state update
        self.c_t = self.f_t * c_prev + self.i_t * self.c_hat_t
        h_t = self.o_t * np.tanh(self.c_t)

        return h_t, self.c_t

    def backward(self, dh_next, dc_next, learning_rate):
        """
        Backprop through time for one timestep.
        dh_next: gradient wrt next hidden state
        dc_next: gradient wrt next cell state
        """

        # Derivatives
        do = dh_next * np.tanh(self.c_t)
        dco = dh_next * self.o_t * (1 - np.tanh(self.c_t) ** 2)
        dc = dco + dc_next

        df = dc * self.c_prev
        di = dc * self.c_hat_t
        dc_hat = dc * self.i_t

        # Gate activation derivatives
        di_input = di * self.i_t * (1 - self.i_t)
        df_input = df * self.f_t * (1 - self.f_t)
        do_input = do * self.o_t * (1 - self.o_t)
        dc_hat_input = dc_hat * (1 - self.c_hat_t ** 2)

        # Gradients wrt parameters
        dW_i = np.outer(di_input, self.x_t)
        dW_f = np.outer(df_input, self.x_t)
        dW_o = np.outer(do_input, self.x_t)
        dW_c = np.outer(dc_hat_input, self.x_t)

        dU_i = np.outer(di_input, self.h_prev)
        dU_f = np.outer(df_input, self.h_prev)
        dU_o = np.outer(do_input, self.h_prev)
        dU_c = np.outer(dc_hat_input, self.h_prev)

        db_i = di_input
        db_f = df_input
        db_o = do_input
        db_c = dc_hat_input

        # Gradients wrt previous hidden and cell states
        dh_prev = (
            self.U_i.T @ di_input
            + self.U_f.T @ df_input
            + self.U_o.T @ do_input
            + self.U_c.T @ dc_hat_input
        )
        dc_prev = dc * self.f_t

        # Parameter updates
        for param, grad in [
            (self.W_i, dW_i),
            (self.W_f, dW_f),
            (self.W_o, dW_o),
            (self.W_c, dW_c),
            (self.U_i, dU_i),
            (self.U_f, dU_f),
            (self.U_o, dU_o),
            (self.U_c, dU_c),
            (self.b_i, db_i),
            (self.b_f, db_f),
            (self.b_o, db_o),
            (self.b_c, db_c),
        ]:
            param -= learning_rate * grad

        return dh_prev, dc_prev


class LSTMNetwork:
    """
    Unrolled LSTM for full sequence modeling.
    seq_len timesteps → final hidden state → linear output → next-day close.
    """

    def __init__(self, input_size, hidden_size, learning_rate=0.001, seq_len=30):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seq_len = seq_len

        self.cell = LSTMCell(input_size, hidden_size)

        # Output layer: maps final hidden state → scalar close prediction
        limit = 1.0 / np.sqrt(hidden_size)
        self.W_y = np.random.uniform(-limit, limit, (1, hidden_size))
        self.b_y = np.zeros((1,))

    def forward(self, X_seq):
        """
        X_seq: (seq_len, input_size)
        Returns predicted close scalar.
        """

        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        self.h_list = []
        self.c_list = []

        # Unroll through time
        for t in range(self.seq_len):
            h, c = self.cell.forward(X_seq[t], h, c)
            self.h_list.append(h)
            self.c_list.append(c)

        # Output layer
        self.y_pred = float(self.W_y @ h + self.b_y)
        return self.y_pred, h, c

    def backward(self, X_seq, y_true):
        """
        Compute gradients for output and LSTM cell using BPTT.
        """
        # Loss derivative
        dy = 2 * (self.y_pred - y_true)

        # Output layer backward
        dW_y = dy * self.h_list[-1]
        db_y = dy

        dh = dy * self.W_y.flatten()
        dc = np.zeros(self.hidden_size)

        # Update output layer
        self.W_y -= self.learning_rate * dW_y.reshape(1, -1)
        self.b_y -= self.learning_rate * db_y

        # Backprop through time
        for t in reversed(range(self.seq_len)):
            dh, dc = self.cell.backward(dh, dc, self.learning_rate)

    def train_step(self, X_seq, y_true):
        """
        One forward + backward pass for a single sequence.
        """
        self.forward(X_seq)
        self.backward(X_seq, y_true)

        loss = (self.y_pred - y_true) ** 2
        return loss
