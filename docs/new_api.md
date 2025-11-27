# Rusty-Coding

After putting lots of time in Rust, and working on other libraries, I am found my style to code.
This library now will be rewritten with somewhat new style that aims to be more readable and maintainable.

The changes in the code base will of course not change the result. The idea is only to use what I have learned in Rust and make the code predictable, explicit and type-aware.
The performance will not be changed either. (However I will try to improve the performance in another way!)

## Examples

### Type-Awareness

When I first started coding I tried to put everything in comments:

```python
@njit
def g(x, n_windows, l, sigma_t):
    """
    Helper function to calculate the approx. confined gaussian window
    as defined in https://doi.org/10.1016/j.sigpro.2014.03.033

    Parameters
    ----------
    x : array
        points at which to calculate the function
    n_windows : int
        length of window in points
    l : int
        n_windows + 1
    sigma_t : float
        parameter of the approx. confined gaussian window (is 0.14)
    """
    ge_e = x - n_windows/2
    ge_d = 2 * l * sigma_t

    sqrt_ge = ge_e / ge_d
    ge = - sqrt_ge*sqrt_ge
    gaus = np.exp(ge)

    return gaus
```

However this makes the code very hard to maintain! what if we change a name of an input? We have to also change the comments. (let's be honest... no one does that.) So I decided to write this function in this way:

```python
@njit
def gaussian_window(x: Tensor,
                    n_windows: int,
                    l: int,
                    sigma_t: float) -> Tensor:
    """
    Approx. confined Gaussian window (see DOI:10.1016/j.sigpro.2014.03.033)
    """

    center = n_windows * 0.5
    denom  = 2.0 * l * sigma_t

    t = (x - center) / denom
    return np.exp(-t * t)
```

(actually right now I don't remember if x was and list or a tensor because back then I used words synonymously, but you get the idea!)

Also the names I used back then for the function were short and needed comments to tell the developer what they actually are. Now I aim to change that.

### Mixing libraries
For some reason I used `numpy.linspace` in my code... I could have stick to the `Torch` function and not mixed liberaries. So I am aiming to change that too!
However I have used `@njit` sometimes and I have to make sure everything will still work together. 

### CI tests
I never did tests back then because I didn't know what they are. I know now and I am aiming to add them too!