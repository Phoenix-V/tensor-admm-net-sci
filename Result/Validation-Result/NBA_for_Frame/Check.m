load('Data_Visualization_0.mat')
[batch,height,width,frame] = size(truth);
psnr_frame = zeros([1 batch*frame]);
ssim_frame = zeros([1 batch*frame]);
for b = 1:batch
    for f = 1:frame
        bf=(b-1)*frame+f; pf=double(pred(b,:,:,f)); of=double(truth(b,:,:,f));
        psnr_frame(bf) = psnr(pf,of,max(of(:)));
        ssim_frame(bf) = ssim(pf,of);
    end
end
mean(psnr_frame)
mean(ssim_frame)