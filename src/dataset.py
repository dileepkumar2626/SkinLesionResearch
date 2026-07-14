import os
import pandas as pd
import torch
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

data_path = r"G:\My Drive\ham10000_data"

print(os.listdir(data_path))
df=pd.read_csv(r"G:\My Drive\ham10000_data\HAM10000_metadata.csv")
print(df.head())
class HAM10000Dataset(Dataset):
  def __init__(self,dataframe,img_dirs,transform=None):
    self.df=dataframe
    self.transform=transform
    self.img_path={}
    for d in img_dirs:
      for fname in os.listdir(d):
        if fname.endswith(".jpg"):
          image_id=fname.replace(".jpg","")
          self.img_path[image_id]=os.path.join(d,fname)
    self.classes=sorted(self.df["dx"].unique())
    self.class_to_idx={cls: i for i,cls in enumerate(self.classes)}
  def __len__(self):
    return len(self.df)
  def __getitem__(self,idx):
    row=self.df.iloc[idx]
    image_id=row["image_id"]
    label=int(self.class_to_idx[row["dx"]])
    img_path = self.img_path.get(image_id)
    if img_path is None:
      raise FileNotFoundError(f"Image {image_id} not found.")
    image=Image.open(img_path).convert("RGB")
    if self.transform:
      image=self.transform(image)
    return image,label
def get_dataloader(batch_size=32):
  unique_leison=df["lesion_id"].unique()
  train_dataset,val_dataset=train_test_split(unique_leison,test_size=0.15,random_state=42)
  train_dataset=df[df["lesion_id"].isin(train_dataset)].reset_index(drop=True)
  val_dataset=df[df["lesion_id"].isin(val_dataset)].reset_index(drop=True)
  print(f"Train: {len(train_dataset)} rows & Val: {len(val_dataset)} Rows")
  train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
  ])
  val_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
  ])
  train_dataset=HAM10000Dataset(dataframe=train_dataset,img_dirs=[r"G:\My Drive\ham10000_data\HAM10000_images_part_1",r'G:\My Drive\ham10000_data\ham10000_images_part_2'],transform=train_transform)
  val_dataset=HAM10000Dataset(dataframe=val_dataset,img_dirs=[r"G:\My Drive\ham10000_data\HAM10000_images_part_1",r'G:\My Drive\ham10000_data\ham10000_images_part_2'],transform=val_transform)
  train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=32,num_workers=2,pin_memory=True)
  val_dataloader=DataLoader(val_dataset,shuffle=False,batch_size=32,num_workers=2,pin_memory=True)
  return train_dataloader,val_dataloader
if __name__=="__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  train_dataloadr,val_dataloader=get_dataloader()