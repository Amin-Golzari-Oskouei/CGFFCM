% This demo implements the CGFFCM algorithm as described in
% A.Golzari oskouei, M.Hashemzadeh, B.Asheghi  and M.Balafar, "CGFFCM: Cluster-weight
% and Group-local Feature-weight learning in Fuzzy C-Means clustering algorithm for color
% image segmentation", Applied Soft Computing, 2021 (submited).
%
% Courtesy of A.Golzari

clc
clear all
close all

%% Load dataset.

%Load image
Img = imread('3096.jpg');

%Load the class.
B=load('class3096.mat');
B=B.class3096;
class=double(reshape(B,[size(B,1)*size(B,2) 1]));

%% Show the image and its Ground truth.

figure(1),imshow(Img),title('Original Image');
% axis off
% img = frame2im(getframe(gca));
% imwrite(Img,'113044.png');

Color_Map=[0.960012728938293,0.867750283531604,0.539798806341941;
    0.00620528530061783,0.456645636880323,0.807705802687295;
    0.16111134052427435,0.0634483119173452,0.321794465524281;
    0.394969976441930,0.662682208014249,0.628325568854317;
    0.0945269059837998,0.794795142853933,0.473068960772494];

Ground_truth = label2rgb(B,Color_Map);
figure(2),imshow(Ground_truth),title('Ground truth Image');
% axis off
% img = frame2im(getframe(gca));
% imwrite(Ground_truth,'class134052.png');

%% Feathre Extract step
fprintf('The feature extraction phase has started ...\n')
X = FeatureExtractor(Img);
[N,d]=size(X);

%% Algorithm parameters.
% for each image, ICA algorithm run and Group Weight values are reported.
% Also, best values for q and landa are listed.
% for get best, rsult we use parameters as fallows:
%==========================================================================
% Image     Weight(G1=V)  Weight(G2=V)  Weight(G3=V)  landa(oo)  q(beta_z)
% 67079          0.0          1              0.0          0.0001      2
% 101027         0.2         0.6             0.2          0.0001      -2
% 118035         0.4         0.6             0.0          0.0010      -2
% 167062         0.4         0.1             0.5          1.0000      2       
% 108073         0.7         0.1             0.2          0.0010      -2  
% 113016         0.7         0.2             0.1          0.0100      -8
% 113044	     0.3         0.6             0.1          0.1000      4
% 134052         0.5         0.4             0.1          0.1000      6       
% 135069         0.1         0.7             0.2          0.0100      -6            
% 238011         0.1         0.1             0.8          0.0001      2	  
% 299091         0.1         0.8             0.1          0.1000      6	
% 3096           0.1         0.7             0.2          0.0001      -10      
% 80099          0.3         0.6             0.1          0.0100      -2
%==========================================================================

k=size(unique(class),1);        % number of clusters.
beta_z = -10;                    % The power value of the feature weight(in paper).
p_init = 0;                     % initial p.
p_max = 0.5;                    % maximum p.
p_step = 0.01;                  % p step.
t_max = 100;                    % maximum number of iterations.
beta_memory = 0.3;              % amount of memory for the weights updates.
Restarts = 1;                   % number of CGFFCM restarts.
fuzzy_degree = 2;               % fuzzy membership degree
v(1,1:3) = 0.1;                 % Weight of group 1
v(1,4:6) = 0.7;                 % Weight of group 2
v(1,7:8) = 0.2;                 % Weight of group 3
G = [1 1 1 2 2 2 3 3];          % Feature Groups (three group 1, 2 and 3)
oo=0.0001;                        % interval (0,1]
landa=oo./var(X);               % the inverse variance of the m-th feature
TF = find(isinf(landa)==1);
if ~isempty(TF)
    for i=1:size(TF,2)
        landa(1,TF(i))=nan;
    end
    aa=max(landa);
    for i=1:size(TF,2)
        landa(1,TF(i))=aa+1;
    end
end

%% Cluster the instances using the CGFFCM procedure.
best_clustering=zeros(1,N);

for repeat=1:Restarts
    fprintf('========================================================\n')
    fprintf('CGFFCM: Restart %d\n',repeat);
    
    %Randomly initialize the cluster centers.
    rand('state',repeat)
    tmp=randperm(N);
    M=X(tmp(1:k),:);
    
    %Execute CGFFCM.
    %Get the cluster assignments, the cluster centers and the cluster variances.
    [Cluster_elem,M,EW_history,W,z]=CGFFCM(X,M,k,p_init,p_max,p_step,t_max,beta_memory,N,fuzzy_degree,d,beta_z,landa,v,G);
    [~,Cluster]=max(Cluster_elem,[],1);
    
    %Meaures
    EVAL = Evaluate(class,Cluster');
    accurcy_ave(repeat)=EVAL(1);
    fm_ave(repeat)=EVAL(2);
    nmi_ave(repeat)=EVAL(3);

    
    if best_clustering ~= 0
        if accurcy_ave(repeat) > accurcy_ave(repeat-1)
            best_clustering = Cluster;
        end
    else
        best_clustering = Cluster;
    end
    
    fprintf('End of Restart %d\n',repeat);
    fprintf('========================================================\n\n')
end

%% Results
% Show the best segmented image
cmap = reshape(best_clustering', [size(double(Img),1) size(double(Img),2)]);
clusteredImage = label2rgb(cmap,Color_Map);
figure(3),imshow(clusteredImage),title('The segmented Image');
% axis off
% img = frame2im(getframe(gca));
% imwrite(clusteredImage,'cluster134052.png');

fprintf('Average accurcy score over %d restarts: %f.\n',Restarts,mean(accurcy_ave));
fprintf('Average F-Measure score over %d restarts: %f.\n',Restarts,mean(fm_ave));
fprintf('Average NMI score over %d restarts: %f.\n',Restarts,mean(nmi_ave));
