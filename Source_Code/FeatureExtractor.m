function FeatureVector=FeatureExtractor(Img)

Img=double(Img)/255;

d=5; %% Size of Window
NumFeatures=8; %% Number of Extracted Features

%% RGB to HSV
imHSV=rgb2hsv(Img);
H=imHSV(:,:,1);
S=imHSV(:,:,2);
V=imHSV(:,:,3);

[rows,cols,~]=size(Img);
FeatureVector=zeros(rows*cols,NumFeatures);

k=2; %% Size of Padding for Image
imGray=rgb2gray(Img);
E_ij=EijCalculator(imGray);    %% Eq(16)

for j=1:3 %% Calculate V_ij in HSV Channels
    
    mu_ij=MuCalculator(imHSV(:,:,j),d,k);       %% Eq(15)
    V_ij=VijCalculator(imHSV(:,:,j),d,k,mu_ij); %% Eq(14)
    H_ij(:,:,j)=1 - (E_ij .* V_ij);                    %% Eq(17)
end

FeatureVector(:,1:3)=reshape(H_ij,[size(H_ij,1)*size(H_ij,2) 3]);

%% Lab Color Space
imLab=rgb2lab(Img);
FeatureVector(:,4:6)=reshape(imLab,[size(imLab,1)*size(imLab,2) 3]);

%% texture components
% GCLM Method
imgray = rgb2gray(Img);
offsets = [0 1; -1 1;-1 0;-1 -1];
NumLevels=5;
[glcm,SI] = graycomatrix(imgray,'Offset',offsets,'Symmetric',true,'NumLevels',NumLevels,'GrayLimits',[]);

% Gabor Filter
imageSize = size(Img);
numRows = imageSize(1);
numCols = imageSize(2);

wavelengthMin = 4/sqrt(2);
wavelengthMax = hypot(numRows,numCols);
n = floor(log2(wavelengthMax/wavelengthMin));
wavelength = 2.^(0:(n-2)) * wavelengthMin;

deltaTheta = 45;
orientation = 0:deltaTheta:(180-deltaTheta);

g = gabor(wavelength,orientation);

gabormag = imgaborfilt(imgray,g);

for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    K = 3;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),K*sigma);
end

X = 1:numCols;
Y = 1:numRows;
[X,Y] = meshgrid(X,Y);
featureSet = cat(3,gabormag,X);
featureSet = cat(3,featureSet,Y);

numPoints = numRows*numCols;
X = reshape(featureSet,numRows*numCols,[]);

X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide,X,std(X));

coeff = pca(X);
feature2DImage = reshape(X*coeff(:,1),numRows,numCols);

FeatureVector(:,7)=reshape(SI,[size(SI,1)*size(SI,2) 1]);
FeatureVector(:,8)=reshape(feature2DImage,[size(feature2DImage,1)*size(feature2DImage,2) 1]);

