def batch(iterable, n=1):
    """Allow to iterate in batch of size n over the given iterable
    
    Args:
        iterable (:iter:): Any iterable with a `len` function.
        n (int, optional): Int specifying the size of the batch.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]