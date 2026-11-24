function Z = fml_mds(D,d)
% Common metric-stress MDS objective for tuning and final fitting.
persistent restart_reported
n = size(D,1);
D = fml_validate_distances(D,n);
validateattributes(d,{'numeric'}, ...
    {'real','finite','scalar','integer','positive','<',n});
try
    Z = mdscale(D,d,'Criterion','metricstress');
catch err
    if ~strcmp(err.identifier,'stats:mdscale:NoSolution')
        rethrow(err)
    end
    % Retain the metric-stress objective when its classical initialization
    % cannot descend. The local restart must not change later CV draws.
    previous_rng = rng;
    restore_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
    rng(0,'twister');
    if isempty(restart_reported)
        warning('FML:MDSRestart', ...
            'Metric-stress MDS needed a deterministic random-start retry.');
        restart_reported = true;
    end
    Z = mdscale(D,d,'Criterion','metricstress','Start','random');
end
if ~isequal(size(Z),[n,d]) || any(~isfinite(Z(:)))
    error('FML:InvalidEmbedding','MDS did not return a finite n-by-d embedding.')
end
end
