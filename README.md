# When an LLM repeatedly rewrite its own output, how quickly does meaning drift, and does it depend on the model, temperature or text type?

As LLM - generated text becomes more and more prolific on the web, we have become concerned about and seen evidence of model collapse as training data increasingly contains
output from earlier models. 

It is logical that this would skew the distributions of text in subsequently trained models, and that this statistical drift would affect successor models.

I am interested in finding out if this applies on the micro-scale, within a single model. Put simply, can a model iterating on its own output, such as an AI agent, catastrophically collapse by overfitting to its training data?

### The experiment: 
