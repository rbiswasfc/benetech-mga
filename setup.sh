hdir=$(pwd)
cd ..

mkdir datasets
mkdir models
mkdir datasets/processed
mkdir datasets/processed/fold_split
mkdir datasets/processed/deps
mkdir datasets/processed/mga_icdar


cd datasets

kaggle competitions download -c benetech-making-graphs-accessible
unzip benetech-making-graphs-accessible.zip -d benetech-making-graphs-accessible
rm benetech-making-graphs-accessible.zip

kaggle datasets download -d conjuring92/mga-fold-split
unzip mga-fold-split.zip -d ./processed/fold_split
rm mga-fold-split.zip

kaggle datasets download -d conjuring92/mga-deps
unzip mga-deps.zip -d ./processed/deps
rm mga-deps.zip

kaggle datasets download -d conjuring92/mga-synthetic
unzip mga-synthetic.zip -d ./processed
rm mga-synthetic.zip

kaggle datasets download -d conjuring92/mga-icdar
unzip mga-icdar.zip -d ./processed/mga_icdar
rm mga-icdar.zip

kaggle datasets download -d conjuring92/mga-pl
unzip mga-pl.zip -d ./processed
rm mga-pl.zip

cd $hdir