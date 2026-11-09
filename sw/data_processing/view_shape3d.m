function view_shape3d(name, dataDir)
% VIEW_SHAPE3D  Plot a synthetic 3D POF sample made by gen_pof3d.py.
%   view_shape3d                       % pof_00000 from sample3d/
%   view_shape3d('pof_00003')
%   view_shape3d('pof_00003', 'out3d')
%
% Generate data first:  python gen_pof3d.py --n 6 --out sample3d
% Same plots as "3D shape reconstruction.ipynb": 3D centerline, camera views,
% rig layout, and triangulation of the GT projections back to 3D.
% All 3D figures can be rotated with the mouse.

if nargin < 1, name = 'pof_00000'; end
if nargin < 2, dataDir = fullfile(fileparts(mfilename('fullpath')), 'sample3d'); end

% ------------------------------------------------------------------ load
% columns: s_norm, X/Y/Z_m (world, z up, clamp at origin), tx/ty/tz (unit tangent),
%          curvature_per_m, twist_per_m, then u/v [px] and z [m] depth in camera 0 and 1
df = readtable(fullfile(dataDir, 'centerlines', [name '.csv']), 'VariableNamingRule', 'preserve');
metas = jsondecode(fileread(fullfile(dataDir, 'meta.json')));
if ~iscell(metas), metas = num2cell(metas); end           % cell when stereo + multiview fields mix
meta = metas{cellfun(@(m) strcmp(m.name, name), metas)};
cams = meta.cameras;

fprintf('%s  %s  L=%.3f m  %s rig  tip force [N] = [%s]\n', name, meta.fiber, ...
    meta.length_m, meta.rig, num2str(meta.tip_force_N', '%.5f '));

X = [df.X_m, df.Y_m, df.Z_m] * 1e3;                      % mm
F = meta.tip_force_N(:)';
kappa = df.curvature_per_m;

% ------------------------------------------------------------------ 3D centerline
figure('Name', [name ' - 3D centerline'], 'Color', 'w');
patch([X(:,1); NaN], [X(:,2); NaN], [X(:,3); NaN], [kappa; NaN], ...
    'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 3, 'DisplayName', 'Centerline');
hold on
plot3(X(1,1), X(1,2), X(1,3), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'DisplayName', 'Clamp');
plot3(X(end,1), X(end,2), X(end,3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Tip');
if norm(F) > 0                                            % cable pull direction at the tip
    v = F / norm(F) * 0.2 * meta.length_m * 1e3;
    quiver3(X(end,1), X(end,2), X(end,3), v(1), v(2), v(3), 0, 'r', 'LineWidth', 1.5, ...
        'MaxHeadSize', 0.5, 'DisplayName', 'Cable force');
end
cb = colorbar; cb.Label.String = 'Curvature (1/m)';
style3d('Synthetic POF Centerline (3D)');

% ------------------------------------------------------------------ camera views
figure('Name', [name ' - camera views'], 'Color', 'w', 'Position', [100 100 1400 650]);
tiledlayout(2, 3, 'TileSpacing', 'compact');
for c = 0:1
    img = imread(fullfile(dataDir, 'images', sprintf('%s_c%d.png', name, c)));
    mask = imread(fullfile(dataDir, 'masks', sprintf('%s_c%d.png', name, c)));
    depth = double(imread(fullfile(dataDir, 'depth', sprintf('%s_c%d.png', name, c)))) * 0.1;  % 0.1 mm units -> mm
    depth(depth == 0) = NaN;

    nexttile; imshow(img); hold on
    % +1: CSV pixel coords are 0-based (OpenCV), MATLAB image coords are 1-based
    plot(df.(sprintf('u%d_px', c)) + 1, df.(sprintf('v%d_px', c)) + 1, 'c-', 'LineWidth', 1);
    title(sprintf('Camera %d - image + GT projection', c));
    nexttile; imshow(mask); title('Mask');
    nexttile; imagesc(depth, 'AlphaData', ~isnan(depth)); axis image off
    cb = colorbar; cb.Label.String = 'Depth (mm)'; title('Depth');
end

% ------------------------------------------------------------------ rig layout
figure('Name', [name ' - camera rig'], 'Color', 'w');
plot3(X(:,1), X(:,2), X(:,3), 'b-', 'LineWidth', 2, 'DisplayName', 'Centerline');
hold on
for c = 1:numel(cams)
    R = cams(c).R; t = cams(c).t(:);
    C = -R' * t * 1e3;                                    % camera centre in world (mm)
    ax = R(3,:)' * 0.3 * norm(C' - mean(X));              % optical axis
    plot3(C(1), C(2), C(3), '^', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', sprintf('Camera %d', c-1));
    quiver3(C(1), C(2), C(3), ax(1), ax(2), ax(3), 0, 'Color', [.5 .5 .5], 'HandleVisibility', 'off');
end
style3d(sprintf('Camera rig (%s)', meta.rig));

% ------------------------------------------------------------------ triangulation check
% later: replace uv0/uv1 with centerlines extracted from the masks
P0 = cams(1).K * [cams(1).R, cams(1).t(:)];
P1 = cams(2).K * [cams(2).R, cams(2).t(:)];
uv0 = [df.u0_px, df.v0_px];
uv1 = [df.u1_px, df.v1_px];
Xrec = triangulate_dlt(P0, P1, uv0, uv1) * 1e3;

figure('Name', [name ' - reconstruction'], 'Color', 'w');
plot3(X(:,1), X(:,2), X(:,3), '-', 'Color', [0 0 1 0.4], 'LineWidth', 4, 'DisplayName', 'Ground truth');
hold on
plot3(Xrec(:,1), Xrec(:,2), Xrec(:,3), 'r--', 'LineWidth', 1, 'DisplayName', 'Triangulated');
style3d('3D Reconstruction vs Ground Truth');
fprintf('max triangulation error: %.2e mm\n', max(abs(Xrec - X), [], 'all'));
end


function style3d(ttl)
title(ttl); xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis equal; grid on; view(3); legend('Location', 'best'); rotate3d on
end


function X = triangulate_dlt(P0, P1, uv0, uv1)
% Linear (DLT) triangulation of matched pixel tracks, (N,2) each -> (N,3).
N = size(uv0, 1);
X = zeros(N, 3);
for i = 1:N
    A = [uv0(i,1) * P0(3,:) - P0(1,:);
         uv0(i,2) * P0(3,:) - P0(2,:);
         uv1(i,1) * P1(3,:) - P1(1,:);
         uv1(i,2) * P1(3,:) - P1(2,:)];
    [~, ~, V] = svd(A);
    X(i,:) = V(1:3,4)' / V(4,4);
end
end
