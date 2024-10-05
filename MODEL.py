import jax.numpy as jnp
from jax import random,jit
from jax import nn
from jaxtyping import Array,Float,PyTree
import matplotlib.pyplot as plt


KEY = random.PRNGKey(42)

class RNN():
    def __init__(self,key,inp,hidden,out):
        self.inp=inp
        self.hidden=hidden
        self.out=out
        self.W_hh = random.normal(KEY,(hidden,hidden))
        self.W_hi = random.normal(KEY,(hidden,inp))
        self.b_h = random.normal(KEY,(hidden,1))
        
        self.W_oh = random.normal(KEY,(out,hidden))
        self.b_o = random.normal(KEY,(out,1))
        
        self.ycaps=None
        self.ht=None
        
    def printsize(self):
        print(self.W_hh.shape)
        print(self.W_hi.shape)
        print(self.b_h.shape)
        print(self.W_oh.shape)
        print(self.b_o.shape)
    
    def forward(self, x, h):
        ht = nn.tanh(jnp.matmul(self.W_hi, x) + jnp.matmul(self.W_hh, h) + self.b_h) 
        out = jnp.matmul(self.W_oh, ht) + self.b_o
        return ht, out

    jit_forward = jit(forward,static_argnums=[0])
        
    def forwardpass(self, x, h0=None, key=random.PRNGKey(0)):
            N = x.shape[0]

            if h0 is None:
                h0 = random.normal(key, (self.hidden, 1))
            
            self.ycaps = jnp.zeros((N, self.out, 1))  # Initialize output container
            self.ht = jnp.zeros((N + 1, self.hidden, 1))  # Initialize hidden state container

            self.ht = self.ht.at[0].set(h0)  # Set the initial hidden state
            
            for i in range(N):
                # Ensure x[i] is reshaped correctly as a column vector
                h0, out = self.jit_forward(x[i].reshape(self.inp, 1), h0)
                self.ht = self.ht.at[i + 1].set(h0)
                self.ycaps = self.ycaps.at[i].set(out.reshape(self.out, 1))  # Ensure output is reshaped correctly

            return self.ycaps, self.ht  # Return predicted outputs and hidden states
            
        