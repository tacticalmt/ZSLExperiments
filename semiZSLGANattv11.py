import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import *
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import os
from torch.optim import RMSprop
from torchvision.utils import make_grid
from pylab import plt
# import Mod.res_block as resblock
import Liblinks.utis as f_s
from Liblinks.sn_lib import *
from Liblinks.resblock import *

tar_path = os.path.abspath('../../dataset/GZSL_for_AWA/Animals_with_Attributes2/')
bat_size = 8
img_size = 256
gpu = True
tr_epoch = 10000
worker = 2
run_mode = 'training'  # 1.training 2.gzsl 3.zsl
# dim=300
noise_size = 128  # 随机变量z的维度，用于输入给生成器
channel = 64
learningr = 0.00001
times_batch = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
version_p = ' v10001a '
back_version = ' v11 '
lr_lambda = 0.005

transform1 = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])  # 可能要换多个transform


# transform2


# transform3


# 数据集载入
class ZSLData(Dataset):
    def __init__(self, data_root, transforms=None, data_mode='training'):
        if data_mode == 'training':
            data_type = 'trainingset'
            task_type = 'trainclasses'

        elif data_mode == 'gzsl':
            data_type = 'testingset'
            task_type = 'gzslclasses'

        elif data_mode == 'zsl':
            data_type = 'testingset'
            task_type = 'testclasses'

        word = f_s.get_name_list(data_root, task_type)  # 获取该任务类型的类名的word list
        att_word = f_s.get_name_list(data_root, 'gzslclasses')  # 获取全部class的word list用以构造attribute字典
        label_dict = f_s.get_dict_name(data_root)  # 类名字典，包含真实类标和类名对应关系
        # print(word)
        # print(att_word)
        # print(label_dict)
        att_np = f_s.get_attri_label(data_root)
        # print(att_np)
        att_dict = f_s.create_name_to_attri(att_word, att_np)  # attribute字典构造
        samp_dir, samp_info = f_s.get_path_info(word, label_dict, data_root, data_type)
        self.image_files, label_true, attribute, train_label = f_s.data_pack_attr(samp_dir, samp_info, att_dict)
        # 把属性向量转成Tensor
        np_train_label = np.array(train_label)
        self.train_label = torch.from_numpy(np_train_label).type(torch.LongTensor)
        np_attribute = np.array(attribute)
        # print(torch.from_numpy(np_attribute))
        tensor_attri = torch.from_numpy(np_attribute)
        # print(tensor_attri[85][11])
        label_np = np.array(label_true)
        label_tensor = torch.from_numpy(label_np).type(torch.LongTensor)  # 转成LongTensor能输入embedding层
        self.transforms = transforms
        self.true_label = label_tensor
        # self.att_label=torch.from_numpy(attribute)#转成Tensor
        self.att_label = tensor_attri

    def __getitem__(self, index):
        try:
            pic_data = self.transforms(Image.open(self.image_files[index]))

        except RuntimeError:
            print(self.image_files[index])
            return self.transforms(Image.open(self.image_files[index - 1])), self.att_label[index - 1], self.true_label[
                index - 1], self.train_label[index - 1]  # ，len(self.image_files)
        else:
            return pic_data, self.att_label[index], self.true_label[index], self.train_label[
                index]  # ，len(self.image_files)

    def __len__(self):
        return len(self.image_files)  # 因为self.images_files是列表，所以能用len来计算大小


# backbone网络
class BackBoneNet(nn.Module):
    def __init__(self, channel=64, n_cls=0):
        super(BackBoneNet, self).__init__()
        self.boneNet = nn.Sequential(OptimizedBlock(3, channel),
                                     Block(channel, channel * 2, downsample=True),
                                     Block(channel * 2, channel * 4, downsample=True),
                                     Block(channel * 4, channel * 8, downsample=True),
                                     Block(channel * 8, channel * 8, downsample=True),
                                     Block(channel * 8, channel * 16, downsample=True),
                                     Block(channel * 16, channel * 16),
                                     nn.ReLU(inplace=True))

    def forward(self, input):
        return self.boneNet(input)


# discriminator模块
# class Discrim(nn.Module):
#    def __init__(self, in_nu, hiddent_nu, out_nu, n_cls=0):  # out_nu为输出维度，判别器维度为1
#        super(Discrim, self).__init__()
#        self.avg = nn.AdaptiveAvgPool2d(1)
#        self.FC = nn.Sequential(SNLinear(in_nu, hiddent_nu),
#                                nn.ReLU(inplace=True)
#                                )
#        self.FD = nn.Sequential(SNLinear(hiddent_nu, out_nu))
#        if n_cls > 0:
#            self.l_y = SNEmbedCLS(n_cls, hiddent_nu)

#    def forward(self, x, y=None):
#        x = self.avg(x)
#        x = x.view(x.size(0), -1)  # 把每个batch变成向量，然后输入全连接层
#        h = self.FC(x)
#        output = self.FD(h)
# print(output)
#        if y is not None:
#            w_y = self.l_y(y)
#            cls_y = torch.mul(w_y, h).sum(dim=1, keepdim=True)  # 要求两个行向量做内积
# print(cls_y)
#            output += cls_y
#        return output


# dicriminator2 号
class Discrim(nn.Module):
    def __init__(self, in_nu, out_nu, n_attr=0):  # out_nu为输出维度，判别器维度为1
        super(Discrim, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(SNLinear(in_nu, in_nu),
                                nn.ReLU(inplace=True)
                                )
        self.FD = nn.Sequential(SNLinear(in_nu, out_nu))
        if n_attr > 0:
            self.l_y = SNLinear(n_attr, in_nu)

    def forward(self, x, y=None):  # y为attri类标
        x = self.avg(x)
        h = x.view(x.size(0), -1)  # 把每个batch变成向量，然后输入全连接层
        # h = self.FC(x)
        output = self.FD(h)
        # print(output)
        if y is not None:
            w_y = self.l_y(y)
            cls_y = torch.mul(w_y, h).sum(dim=1, keepdim=True)  # 要求两个行向量做内积
            # print(cls_y)
            output += cls_y
        return output


# 回归器部分
class Regressor(nn.Module):
    def __init__(self, in_nu, hidden_nu, out_nu=85):  # out_nu为特征维数
        super(Regressor, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(SNLinear(in_nu, hidden_nu),
                                nn.ReLU(inplace=True),
                                SNLinear(hidden_nu, hidden_nu),
                                nn.ReLU(inplace=True),
                                SNLinear(hidden_nu, out_nu),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        output = self.FC(x)
        return output


# 生成器网络
class Generat(nn.Module):
    def __init__(self, channel=64, dim_z=128, bottom_width=4, n_cls=0):
        self.bottom_width = bottom_width
        self.dim_z = dim_z
        self.num_classes = n_cls
        super(Generat, self).__init__()
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * channel * 16)
        # print(self.l1)
        self.b1 = GBlockAttr(channel * 16, channel * 16, upsample=True, num_attr=n_cls)
        self.b2 = GBlockAttr(channel * 16, channel * 8, upsample=True, num_attr=n_cls)
        self.b3 = GBlockAttr(channel * 8, channel * 8, upsample=True, num_attr=n_cls)
        self.b4 = GBlockAttr(channel * 8, channel * 4, upsample=True, num_attr=n_cls)
        self.b5 = GBlockAttr(channel * 4, channel * 2, upsample=True, num_attr=n_cls)
        self.b6 = GBlockAttr(channel * 2, channel, upsample=True, num_attr=n_cls)
        self.bn1 = nn.BatchNorm2d(channel)
        self.con1 = nn.Conv2d(channel, 3, kernel_size=3, stride=1, padding=1)
        self.actf = nn.ReLU(inplace=True)
        self.tanhf = nn.Tanh()

    def forward(self, z, y=None, bat_size=10, **kwargs):
        if z is None:
            z = torch.randn(bat_size, self.dim_z).cuda()
        if y is None:
            np_y = np.random.randint(2, size=(bat_size, self.num_classes))
            y = torch.from_numpy(np_y).type(torch.FloatTensor).cuda()
        x = self.l1(z)
        x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)
        # print(x.size())
        x = self.b1(x, y, **kwargs)
        x = self.b2(x, y, **kwargs)
        x = self.b3(x, y, **kwargs)
        x = self.b4(x, y, **kwargs)
        x = self.b5(x, y, **kwargs)
        x = self.b6(x, y, **kwargs)
        x = self.bn1(x)
        x = self.actf(x)
        x = self.con1(x)
        x = self.tanhf(x)
        return x


def weight_init(m):  # 初始化函数
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 or class_name.find('Linear') != -1:
        m.weight.data.normal_(0, 0.02)
    # elif class_name.find('Norm') != -1:
    #    m.weight.data.normal_(1.0, 0.02)


# 网络对象定义
back_bone = BackBoneNet(channel=channel)
discriminator = Discrim(channel * 16, 1, n_attr=85)
generator = Generat(channel, dim_z=noise_size, n_cls=85)  # channel,noise_size,4,0
regressor_net = Regressor(channel * 16, 1024, 85)
loss = nn.MSELoss()

back_bone.apply(weight_init)
discriminator.apply(weight_init)
generator.apply(weight_init)
regressor_net.apply(weight_init)

# 训练配置
# optimizerD=RMSprop(discriminator.parameters(),lr=learningr)
# optimizerR=RMSprop(regressor_net.parameters(),lr=learningr)
# optimizerB=RMSprop(back_bone.parameters(),lr=learningr)
# optimizerG=RMSprop(generator.parameters(),lr=learningr)
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learningr)
optimizerR = torch.optim.Adam(regressor_net.parameters(), lr=learningr)
optimizerB = torch.optim.Adam(back_bone.parameters(), lr=learningr)
optimizerG = torch.optim.Adam(generator.parameters(), lr=learningr)

one = torch.ones([1], dtype=torch.float)
mone = one * -1
dataset = ZSLData(tar_path, transforms=transform1, data_mode=run_mode)
data_container = DataLoader(dataset, bat_size, shuffle=True, num_workers=worker)

fix_noise = torch.randn(bat_size, noise_size)

if gpu:
    back_bone.cuda()
    discriminator.cuda()
    generator.cuda()
    regressor_net.cuda()
    fix_noise = fix_noise.cuda()
    mone = mone.cuda()

# 训练


for i_epoch in range(tr_epoch):
    for i_data, data in enumerate(data_container):
        x_image = data[0]  # 图片
        x_label = data[1].type(torch.FloatTensor)  # 训练用的回归类标   要求是Tensor
        # x_ground_truth = data[2]  # 真实类标  可能可以用转化成LongTensor来做输入  类标超界，所以减一
        # x_train_label = data[3]
        # x_ground_test=x_ground_truth-1
        amount_data = len(data_container)  # data[3]#最大样本数
        noise = torch.randn(x_image.size(0), noise_size)  # 随机生成噪声
        # print(noise.size())
        print(i_data)

        if gpu:
            x_image = x_image.cuda()
            x_label = x_label.cuda()
            noise = noise.cuda()
            # x_ground_truth = x_ground_truth.cuda()
            # x_train_label = x_train_label.cuda()

        out_back = back_bone(x_image)  # 输出特征
        output_d = discriminator(out_back, x_label)  # 判别器网络
        output_regr = regressor_net(out_back)  # 回归器网络
        # 生成样本的输出
        # print(noise.size())
        fake_img = generator(noise, x_label).detach()
        out_back_g = back_bone(fake_img)
        output_g = discriminator(out_back_g, x_label)  # 输入类标,x_ground_truth
        # 计算loss
        regr_loss = loss(output_regr, x_label)  # x_label类型？
        gan_loss = output_g - output_d
        # print(gan_loss)
        overall_loss = gan_loss.mean() + lr_lambda * regr_loss
        print(overall_loss)
        # 优化
        # optimizerD.zero_grad()
        # optimizerR.zero_grad()
        # optimizerB.zero_grad()
        # 异常抛出
        # try :
        #    torch.backends.cudnn.enabled = False
        #    overall_loss.backward()
        # optimizerD.step()
        # optimizerR.step()
        # optimizerB.step()

        # except RuntimeError:
        #    print('skip this batch')
        #    continue

        # else:
        #    optimizerD.step()#更新判别子网
        #    optimizerR.step()#更新回归子网
        #    optimizerB.step()#更新骨干网
        # overall_loss.backward()
        # optimizerD.step()
        # optimizerR.step()
        # optimizerB.step()
        #
        #
        if i_data == 0:
            optimizerD.zero_grad()
            optimizerR.zero_grad()
            optimizerB.zero_grad()
        # torch.backends.cudnn.enabled = False
        overall_loss.backward()
        if (i_data + 1) % times_batch == 0 or i_data == (len(data_container) - 1):
            optimizerD.step()
            optimizerR.step()
            optimizerB.step()

            optimizerD.zero_grad()
            optimizerR.zero_grad()
            optimizerB.zero_grad()

        # if (i_data+1)%3==0:
        #    optimizerD.zero_grad()
        #    optimizerR.zero_grad()
        #    optimizerB.zero_grad()

        print(i_data)
        if (i_data + 1) % (5 * times_batch) == 0:
            for b_i_gan in range(times_batch):
                if b_i_gan == 0:
                    generator.zero_grad()

                noise.data.normal_(0, 1)
                fake_pic = generator(noise, x_label)
                out_back_tg = back_bone(fake_pic)
                output_tg = discriminator(out_back_tg, x_label).mean().view(1)  # ,x_ground_truth
                # torch.backends.cudnn.enabled = False
                output_tg.backward(mone)

            # generator.zero_grad()
            # noise.data.normal_(0,1)
            # fake_pic=generator(noise)
            # out_back_tg=back_bone(fake_pic)
            # output_tg=discriminator(out_back_tg,x_ground_truth).mean().view(1)
            # print(output_tg)
            # output_tg.backward(mone)

            optimizerG.step()
            # output=discriminator(fake_pic).mean().view(1)

    if (i_epoch + 1) % 2 == 0:
        fake_u = generator(fix_noise, bat_size=bat_size)
        imgs = make_grid(fake_u.data * 0.5 + 0.5).cpu()  # CHW
        # plt.imshow(imgs.permute(1,2,0).numpy())
        outprint = tf.to_pil_image(imgs)
        outprint.save(
            './results/' + back_version + str(i_epoch + 1) + ' th ' + version_p + ' epoch attrisemizslGanGAN.png')

    if (i_epoch + 1) % 50 == 0:
        torch.save(back_bone.state_dict(),
                   './models/' + back_version + str(i_epoch + 1) + ' th ' + version_p + ' attrisemiGAN_backbone.pkl')
        torch.save(discriminator.state_dict(),
                   './models/' + back_version + str(
                       i_epoch + 1) + ' th ' + version_p + ' attrisemiGAN_discriminator.pkl')
        torch.save(generator.state_dict(),
                   './models/' + back_version + str(i_epoch + 1) + ' th ' + version_p + ' attrisemiGAN_generator.pkl')
        torch.save(regressor_net.state_dict(),
                   './models/' + back_version + str(
                       i_epoch + 1) + ' th ' + version_p + ' attrisemiGAN_regressor_net.pkl')
