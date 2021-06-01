"""
Knowledge Distillation Super Resolution

지식 증류 기법을 Super Resolution 딥러닝 네트워크를 위한 모듈

Writer : KHS0616
Ver : 0.0.1
Last Update : 2021-05-18
"""
import torch
from torch.optim.lr_scheduler import StepLR
from feature_transformation import *

class KDS():
    """ Knowledge Distillation SuperResolution 메인 클래스 """
    def __init__(self):
        # 측정 척도 초기화
        self.loss, self.psnr, self.ssim, self.lpips, self.timesec = 0, 0, 0, 0, 0

    def setDevice(self, device):
        """ Device CPU/GPU 등록 """
        # 설정된 Device 등록
        self.device = device

    def setTeacher(self, model):
        """ 선생님 모델 등록 """
        # 모델 등록 pretrained 모델 존재하면 weight 불러오고 입력해야한다
        self.teacher_model = model
        print("선생님 모델 등록 완료")

    def setStudent(self, model):
        """ 학생 모델 등록 """
        # 모델 등록 pretrained 모델 존재하면 weight 불러오고 입력해야한다
        self.student_model = model
        print("힉생 모델 등록 완료")

        # Optimizer 설정
        self.setOptimizer()

    def setOptimizer(self, optim=None):
        """ Optimizer 등록 """
        if optim is None:
            self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=0.0001)
        else:
            self.optimizer = optim

        # 스케줄러 등록
        self.scheduler = StepLR(self.optimizer, step_size=150, gamma=0.1)

    def setLoss(self, SA=True):
        """ Loss 선택 및 등록 """
        # Loss 리스트 선언
        self.losses = []
        
        # Loss 항목 추가
        if SA:
            self.losses.append("1*SA")

        # Loss 생성
        self.criterion = Loss(self.losses, self.device)

    def changeLoss(self, name, status):
        """ Loss 구성요소 변경 """
        pass

    def getLoss(self):
        """ 현재 Loss 반환 """
        return self.loss

    def getScheduler(self):
        """ Scheduler 반환 """
        return self.scheduler

    def train(self, lr, hr):
        """ KD 기법 Train Iterator """
        # 데이터 Device 등록
        lr = lr.to(self.device)
        hr = hr.to(self.device)

        # 학생 모델 이미지 통과
        student_fms, student_sr = self.student_model(lr)

        # 선생님 모델 이미지 통과
        with torch.no_grad():
            teacher_fms, teacher_sr = self.teacher_model(lr)
            teacher_sr = teacher_sr.to(self.device)

        # feature map 담을 리스트 선언
        aggregated_student_fms = []
        aggregated_teacher_fms = []

        # loss에 따른 feature map transformation
        if len(self.losses) > 0:
            for feature_loss in self.losses:
                if "SA" in feature_loss:
                    aggregated_student_fms.append([spatial_similarity(fm) for fm in student_fms])
                    aggregated_teacher_fms.append([spatial_similarity(fm) for fm in teacher_fms])

        # loss 측정 및 저장
        self.loss = self.criterion(student_sr, teacher_sr, hr, aggregated_student_fms, aggregated_teacher_fms)

        # 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산한다.
        self.loss.backward()

        # Optimize
        self.optimizer.step()

    def saveStudentModel(self, path):
        """ Student Model 저장 메소드 """
        torch.save(self.student_model.state_dict(), path)

class Loss(torch.nn.modules.loss._Loss):
    """ Loss 클래스 """
    def __init__(self, feature_loss_list, device):
        super(Loss, self).__init__()
        self.n_GPUs = 1
        # Loss를 담을 리스트 선언
        self.loss = []
        self.feature_loss_module = torch.nn.ModuleList()
        
        # feature_loss 사용여부 설정
        if len(feature_loss_list) > 0:
            self.feature_loss_used = 1
        else:
            self.feature_loss_used = 0

        # SR Loss Weight 설정
        DS_weight = 1 - 0.5
        TS_weight = 0.5

        # Sr Loss 등록
        self.loss.append({'type': "DS", 'weight': DS_weight, 'function': torch.nn.L1Loss()})
        self.loss.append({'type': "TS", 'weight': TS_weight, 'function': torch.nn.L1Loss()})

        # feature loss 등록
        if self.feature_loss_used == 1:
            for feature_loss in feature_loss_list:
                weight, feature_type = feature_loss.split('*')
                l = {'type': feature_type, 'weight': float(weight), 'function': FeatureLoss(loss=torch.nn.L1Loss())}
                self.loss.append(l)
                self.feature_loss_module.append(l['function'])
            
        # Total Loss 등록
        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        
        # Device 설정
        self.feature_loss_module.to(device)

    def forward(self, student_sr, teacher_sr, hr, student_fms, teacher_fms):
        # DS Loss
        DS_loss = self.loss[0]['function'](student_sr, hr) * self.loss[0]['weight']
        
        # TS Loss
        TS_loss = self.loss[1]['function'](student_sr, teacher_sr) * self.loss[1]['weight']
        
        loss_sum = DS_loss + TS_loss
        
        if self.feature_loss_used == 0:
            pass
        elif self.feature_loss_used == 1:
            assert(len(student_fms) == len(teacher_fms))
            assert(len(student_fms) == len(self.feature_loss_module))
            
            for i in range(len(self.feature_loss_module)):
                feature_loss = self.feature_loss_module[i](student_fms[i], teacher_fms[i])
                loss_sum += feature_loss

        return loss_sum

    # def save(self, apath):
    #     torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
    #     torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    # def load(self, apath, cpu=False):
    #     if cpu:
    #         kwargs = {'map_location': lambda storage, loc: storage}
    #     else:
    #         kwargs = {}

    #     self.load_state_dict(torch.load(
    #         os.path.join(apath, 'loss.pt'),
    #         **kwargs
    #     ))
    #     self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
    #     for l in self.feature_loss_module:
    #         if hasattr(l, 'scheduler'):
    #             for _ in range(len(self.log)): l.scheduler.step()

class FeatureLoss(torch.nn.Module):
    """ Feature Loss 클래스 """
    def __init__(self, loss=torch.nn.L1Loss()):
        super(FeatureLoss, self).__init__()
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)        
        length = len(outputs)
        tmp = [self.loss(outputs[i], targets[i]) for i in range(length)]
        loss = sum(tmp)
        return loss