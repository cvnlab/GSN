function ix = deterministic_randperm(n, seed)
    if nargin < 2
        seed = 42;
    end
    rng(seed, 'twister');
    ix = randperm(n);
end
