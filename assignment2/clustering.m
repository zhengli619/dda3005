function [c, Z] = clustering(X, Z0, opts)

%--------------------------------------------------------------------------
% INPUT
%--------------------------------------------------------------------------
%
%   - X     (n x m) matrix containing the m different data points.
%   - Z0    (n x k) initial points for the clusters (will be estimated by 
%                   Z0 = rand(n,3) if left empty).
%   - opts  field of options
%               .iter       number of maximum iterations (default = 100).
%               .display    print total distances (default = true).
%
%--------------------------------------------------------------------------

[n, m] = size(X);

%--------------------------------------------------------------------------
% PROCESS INPUT AND OPTIONS
%--------------------------------------------------------------------------

if nargin <= 1
    k = 3;
    Z = randn(n, k);
    opts.iter = 100;
    opts.display = true;
elif nargin <= 2
    k = size(Z0, 2);
    Z = Z0;
    opts.iter = 100;
    opts.display = true;
else
    k = size(Z0, 2);
    Z = Z0;
    if ~isfield(opts, 'iter')
       opts.iter = 100;
    end
    if ~isfield(opts, 'display')
       opts.display = true;
    end
end

sX = repmat(X, k, 1);

%--------------------------------------------------------------------------
% MAIN LOOP
%--------------------------------------------------------------------------

for i = 1:opts.iter
    % calculate distances
    sZ = abs(sX - reshape(Z(:), [], 1)).^2;
    sZ = sqrt(sum(reshape(sZ, n, m * k)));
    % select minimum distances
    sZ = reshape(sZ, k, m);
    [mZ, c] = min(sZ);

    % calculate new means
    for j = 1:k
        ind = (c == j);
        Z(:, j) = mean(X(:, ind), 2);
    end
end

% output
if opts.display
    s1 = sum(sum(mZ.^2));
    s2 = s1 / (m * k);
    fprintf(1, 'Total sum of distances = %5.2f; Weighted sum of distances = %1.6e\n', s1, s2);
end

end