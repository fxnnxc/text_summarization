# codes 

from 2021 01 19 archive, just copy the code below

```python
# model.py
alpha = 0.4
x_z *= alpha
decrease = 1
for i in range(y.size(1)):
    y[:,[i],:] += x_z * decrease
    decrease *= 0.7
```

```python
# sequence_generator.py

# -- encoder part
encoder_out = encoder_outs[i]
latent_z = model.get_latent_z(encoder_out['encoder_out'][0])

alpha = 0.4
decreasing = 0.7 **(tokens.size(1)-1)
latent_vocab = alpha * decreasing * latent_z

## -- decoder addition part

if not model.sep_softmax:
    decoder_prob = decoder_out[0] + latent_vocab
    decoder_out = (decoder_prob, decoder_out[1])

```
