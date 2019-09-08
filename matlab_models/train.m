xb1 = double(xb1);
W1 = exp(-pdist2(xb1,xb1));
W2 = exp(-pdist2(xb2,xb2));
W12 = generateWeight3(xb1',xb2',3);

[map1,map2] = wmapGeneralTwo(xb1', xb2', W1, W2, W12, 0.12, 0.1,size(xb1,2));

clear W1 W2 W12;

new_image = map1'*xb1';
new_word = map2'*xb2';

acc = 0;
for i=1:size(new_image,2)
    diff_matrix = pdist2(new_image(:,i)',new_word');
    [C,I] = min(diff_matrix);
    %[labels1(:,i),labels2(:,I)]
    if labels1(i,1)==labels2(I,1)
        acc = acc + 1;
    end
end

clear diff_matrix;

acc = acc/size(xb1,1);
acc

save mnist_mawc_noisy_6.mat new_image new_word xb1 xb2 labels1 labels2
  
clear all

