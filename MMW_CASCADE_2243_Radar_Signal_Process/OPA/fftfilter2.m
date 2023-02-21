function filterwave = fftfilter2(ffty,fs,downnum,upnum)
%% FFT，滤波，IFFT，取实部

lenffty = length(ffty);
fftfs = 1/fs;
yfft = fft(ffty);
downfreq = round(downnum*lenffty*fftfs);
upfreq = round(upnum*lenffty*fftfs);
yfft(1:downfreq)= 0;
yfft(end-downfreq+1:end)= 0;
yfft(upfreq:end-upfreq+1)= 0;
filterwave = real(ifft(yfft));





