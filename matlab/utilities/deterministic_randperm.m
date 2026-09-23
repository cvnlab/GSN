function ix = deterministic_randperm(n, seed)
    if nargin < 2
        seed = 42;
    end
    % Use a local stream so the caller's global RNG state is untouched.
    % 'mt19937ar' with this seed yields the same permutation as
    % rng(seed,'twister'); randperm(n).
    s = RandStream('mt19937ar', 'Seed', seed);
    ix = randperm(s, n);
end
