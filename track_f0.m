function [f0, tt] = track_f0(wave, ws, voicingThresh, silenceThresh)
% Function [f0, tt] = track_f0(wave, ws, voicingThresh, silenceThresh) use 
% adaptive least squares (ALS) algorithm to find the fundamental frequency of wave at 
% different times.
%
% Input
%      wave: the sound wave data (assuming a sampling frequency of 44100Hz)
%      ws: analysis window size (in seconds). A good choice is 0.04s 
%      voicingThresh: cost must be below this threshold to be considered as
%           a good sinusoidal fit. For Edinburgh data, it was chosen at
%           0.0825
%      silenceThresh: energy of the signal must be above this threshold to
%           be considered as voiced region. For Edinbugh data, it was
%           chosen at 1e2. The silenceThresh is more or less determined by
%           the energy level, therefore should be scaled accordingly to the
%           data being examined.
%
%
% Returns:
%      f0: the estimated fundamental frequency
%      tt: the time stamp (in seconds) when the fundamental frequency is
%      estimated
%
% ALS algorithm: 
%    http://www.cis.upenn.edu/~lsaul/papers/voice_nips02.pdf
%
% written by {feisha, burgoyne}@cis.upenn.edu, adapted from L. Saul's code.
%

% SAMPLING FREQUENCY AND DECIMATION RATE (USED TO SPEED UP PROCESSING)
fs = 44100; decrate = 12;  
% fs = 22050; decrate = 6;  % Alternative params settings

sr = fs / decrate; % new sampling rate after decimation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILTERS
persistent bf af fMin fMax nBand bl; % CLEAR ALL IF YOUR SAMPLING RATE IS 
                                     % CHANGED EXTERNALLY
if (isempty(bf))
    if rem(fs, decrate) ~=0
        error('The sampling rate is NOT the multiple of the decimation rate!');
    end
    % 
    fMin = [50 71 100 141 200 283 400 533];
    fMax = [75 107 150 212 300 425 600 800];
    %
    nBand = length(fMin);
    order = 4; 
    ripple = 0.5;
    bf = zeros(nBand,2*order+1);
    af = zeros(nBand,2*order+1);
    for band=1:nBand
        passBand = [fMin(band)/2 fMax(band)]/(sr/2);
        [bf(band,:),af(band,:)] = cheby1(order,ripple,passBand);
    end;
    bl = fir1(50, 1000/(fs/2)); % LOW PASS FOR DECIMATION

end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRACK
wave = reshape(wave,1,length(wave));
envA = filtfilt(bl, 1, wave);
envA = envA(1:decrate:end);
envB = max(0, envA); 

nFrame = length(envB);
freqs = zeros(nBand,nFrame); costs = zeros(nBand,nFrame);
for band=1:nBand
    sineWave = filtfilt(bf(band,:),af(band,:),envB);
    [freqs(band,:),costs(band,:)] = ...
        TrackBand(sineWave,sr, ws,fMin(band));
end;
[cost,indx] = min(costs,[],1);

% PITCH
pitch = zeros(size(cost));
for band=1:nBand
  pitch(indx==band) = freqs(band,indx==band);
end;

% VOLUME
volume = filter(ones(1,decrate)/decrate,1,wave.*wave);
volume = volume(1:decrate:end);

% VOICED/UNVOICED DETERMINATION
voiced = find(cost<voicingThresh  & volume>silenceThresh);
f0 = zeros(size(pitch));
f0(voiced) = pitch(voiced);

% ALS fitting uses preceding wave data to compose an analysis window
% therefore, we shift half of the window and interpolate
tt = [0:nFrame-1]/sr- ws/2; uu = [0:nFrame-1]/sr; 
f0 = interp1(tt,f0,uu,'nearest');
tt = uu;
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit wave data to a sinusoid, returning estimated frequencies and
% heuristic fitting costs 
function [f0,cost] = TrackBand(xx,samplingRate, windowSize,minF0)

% SINUSOIDAL FIT
nn = ceil(windowSize*samplingRate);   % ANALYSIS WINDOW SIZE
mm = [2*xx(1) xx(1:end-2)+xx(3:end) 2*xx(end)]; % XX(n-1) + XX(n+1)
xm = cumsum(xx.*mm);                  
m2 = cumsum(mm.*mm);                  
x2 = cumsum(xx.*xx);                  
xm = xm-[zeros(1,nn) xm(1:end-nn)];   % NUMERATOR OF EQ.(4)
m2 = m2-[zeros(1,nn) m2(1:end-nn)];   % DENOMINATOR OF EQ.(4)
x2 = x2-[zeros(1,nn) x2(1:end-nn)];   % XX(n) squared in the window
aa = xm./(m2+realmin);
aa(m2==0) = 0.5;

% BELOW MINIMUM FREQUENCY?
minP = 2*pi*minF0/samplingRate;
aa(abs(aa)<0.5/cos(minP)) = 0.5;

% PITCH
f0 = acos(0.5./aa) * samplingRate / (2*pi);
f0 = min(f0,samplingRate-f0);
pp = 2*pi*f0./samplingRate;
sp = sin(pp);
cp = cos(pp);

% COST
cost = sqrt(abs(x2+aa.*aa.*m2-2*aa.*xm)./abs(m2+realmin));
cost = cost.*(cp.*cp*samplingRate)./(pi*sp+realmin);
cost = cost./(f0+realmin);
cost(f0==0) = inf;
cost(m2==0) = inf;
cost(x2==0) = inf;
return;