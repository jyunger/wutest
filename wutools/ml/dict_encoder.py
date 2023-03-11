"""Some thoughts on encoding an event dict to R^N

Let's assume the dict keys are strings and values are possibly nested dicts or atomic types.

One approach is to flatten the nested dict and only try to encode that.
Encoding a simple str -> str, int, float dict might go like this:
    * First step, encode key:value pairs, at least for simple types
    1. Learn to encode the sets of keys. I say sets because maybe 'foo' always appears with 'bar' as a key so you can
        get more compression. Also, you could possibly learn that 'fooo' is a typo of 'foo' if 'bar' is present. The
        decoder must decode to the same number of keys.

Ideas:
1. The keys in a flattened event dict are correlated with the value for the 'event_type' key. How do we learn something
    like that? S = [(k, v)]. P['foo' is a key | ('event_type', 'foo_event') in S]. How do you learn that is important?
1. Here are some relationships worth learning:
    a. type(k, T) = P[type(v) = T | (k, v) in S]
    b. exist(k, k1, v1) = p[k in S | (k1, v1) in S]
1. Some integer values are categoricals, some are ordinals, some are ids. Can you learn which is which? What does that
    mean?
1. How big is the space of events? Can encoding/decoding really de-noise the data?
    A. What are the potential noises in an event dict?
        i. missing or extra keys. How do you know you have a missing or extra key? Some keys should be present for all
            events. Some depend on the value of the event_type key. How do you learn that?
        ii. typos in keys
        iii. bad values
1. Thoughts on encoding key, value pairs:
    * This seems doable. This is just a map from (str, [str, int, float]) -> R^N
        * f(k, v) = f_k(v) (currying)
        * Since k is discrete the set of possible f_k is also discrete
        * f_k can probably be rewritten as f_k(type(v), v)
        * curry again, to get f_k_str, f_k_int, f_k_float
        * f_k_str(s) completely defined by a dict d s.t. f_k_str(s) = d[s].
            * How to limit the parameters in d?
            * Reducing the number of parameters means forming equivalence classes. You need some parameters for forming
                equivalence classes and then a parameter for each equivalence class
                d[s] = d'[equivalence_class(s)]
            * So, what strings belong together in an equivalence class? How would you learn that?
            * If you had a known vocabulary and it was smaller than the number of allowed parameters, you could just
                make the dict and use the remaining parameters to control what to do for previously unseen strings.
                * You probably want to choose the value for the nearest neighbor weighted by frequency, but it probably
                    depends on how you define the cost function for the encode->decode composition.
    * Decoding is a map from R^N to str.
        * Since R^N has higher cardinality, it is possible to perfectly encode/decode.
        * But how do you parameterize?
        * You can do the inverse of one-hot:
            * g_i: R^N -> R^A where A = size of alphabet + 1 (for None)
            * g_hat(x) = (argmax(g_1(x), argmax(g_2(x), ..., argmax(g_V(x)) where V is max word length
            * g(x) = takewhile(g_hat(x), lambda c: c!=None)
        * You can do the inverse of a recurrent:
            * g_hat: R^N -> A x R^N
            * def l(x): while True: c, x = g_hat(x); if c is None: break; else yield c
            * g(x) = ''.join(l(x))
    * Cost function
        * We want to find the f and g that minimizes something in relation to the dataset.
        * To do this, maybe we want g to return a distribution on strings and have the loss be -log_likelihood(s).
            * This will give low loss to transpositions and replacements but very high cost to insertion and deletions.
                * Maybe interpret None to mean no char. Then, if p_i[c] is the prob of c at slot i,
1. Thoughts on stream learning
    * The idea is to keep updating the ML weights as new info comes in rather than a model that learns from a chunk
        of data one time. One approach involves creating and updating a "sketch" of the previous data and using it to
        guide the updating of the model parameters based on new data. This involves estimating how adding the new data
        to the historical data would change the optimization. I think this might require too much knowledge of the
        fitted function to be done analytically. How does a brain do it? Sort of the coefficients decay with time and
        are strengthened by firing.

"""

def items_flat(d):
    for k, v in d.items():
        k = (k,)
        if isinstance(v, dict):
            for ksub, vsub in items_flat(v):
                yield k+ksub, vsub
        else:
            yield k, v

def join_key(dict_items, sep):
    for k, v in dict_items:
        yield sep.join(str(x) for x in k), v


