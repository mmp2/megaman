% generates the test data used by test_laplacian.py
% LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE
%

%addpath /mnt/costila/speclust/code-dominique-rmetric/
addpath /mnt/costila/mmp/research/spectral/dominique-epsilon/EpsilonDemo/

outfroot = 'testmegaman_laplacian'
rad = 0.2;
renormlam = 1.5;   % renormalization exponent
opts.lam = renormlam;
n = 200;
seed  = 36;
rand( 'seed', seed );
xx1 = rand( 1, n );
xx2 = rand( 1, n );
xx3 = sin( 2*pi*xx1).*sqrt(xx2);

xx = [ xx1; xx2; xx3 ];

epps = rad*rad;
[ A, S ] = similarity( xx', epps );
norms = {'geometric', 'unormalized', 'randomwalk', 'symmetricnormalized', 'renormalized' };
names = {'geom', 'unnorm', 'rw', 'symnorm', 'reno1_5' };

for ii = 1:length( norms );
    disp( norms{ ii } )
    opts.lapType = norms{ ii };
    [ L, phi, lam, flag ] = laplacian( A, 2, epps, opts );
    eval( [ 'L' names{ ii } '=L;']);
    eval( [ 'phi' names{ ii } '=phi;']);
    eval( [ 'lam' names{ ii } '=lam;']);
end;

[G, VV, LL, Ginv ] = rmetric( Lgeom, phigeom, 2, 0 );

rad
num2str_(rad)
renormlam
num2str_(renormlam)
outfname = [ outfroot '_rad' num2str_(rad) '_lam' num2str_(renormlam) '_n' num2str( n ) '.mat' ]

save( outfname )
