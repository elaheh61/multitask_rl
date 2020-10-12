import torch
import torch.nn as nn

seed = 0

def init_random(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Joint_Model(nn.Module):
    def __init__(self,input_size, output_size,tasks, hiddens=None, activation='relu',**kwargs):
        super(Joint_Model,self).__init__()
​        self.tasks = tasks
        # model = eval(model_name)
        # model = model(pretrained=True)
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        self.base= [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            self.base.append(linear_init(nn.Linear(i, o)))
            self.base.append(activation())
        # layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        # self.base= nn.Sequential(*list(model.children())[:-1])
        # last_layers = [linear_init(nn.Linear(hiddens[-1], output_size)) * len(tasks)]
        self.branches = {}
        for i in range(len(tasks)):
            self.branches[task[i]] = linear_init(nn.Linear(hiddens[-1], output_size))
​
    def forward(self, x):
        for l in self.base:
             = l(x)
        # x = self.base(x)
        # x = #F.avg_pool2d(x,x.size()[2:]) #
        # f = x.view(x.size(0),-1)
        x1 = torch.flatten(x, 1)
​
        clf_outputs = {}
        for k, v in self.branches.items():
            clf_outputs[k] = v(x1)

        # clf_outputs["fc_target"] = self.fc_target(x1)
​
        return clf_outputs
​
def init_device():
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
​
device = init_device()

def make_env():
    return sunblaze_envs.make(env_name)

env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
env.seed(seed)
env = ch.envs.Torch(env)
meta_bsz = 4
tasks = ["source_"+str(i) for i in range(meta_bsz)]
tasks.append("target")
meta_learner = Joint_Model(env.state_size, env.action_size, tasks)
objective = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for task_config in tqdm(env.sample_tasks(meta_bsz+1)):  # Samples a new config
    env.set_task(task_config)
    env.reset()
    task = ch.envs.Runner(env)

    train_episodes = task.run(meta_learner, episodes=adapt_bsz)
    # loss = maml_a2c_loss(train_episodes, meta_learner, baseline, gamma, tau)
    # print("loss = ", loss)
    # iteration_loss += loss
    # iteration_reward += train_episodes.reward().sum().item() / adapt_bsz
    if(test == False):
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("end: ", meta_learner.context_params)


source_trainloader = None  #dataloader containing source data
# ​target_train_loader = torch.utils.data.DataLoader(train_target, batch_size=target_batch_size, shuffle=True, num_workers=2)
target_trainloader = None  #dataloader containing target data
##### define Resnet18 with two fully connected layer on top: target has 2 classes and source has 5 classes
# model.to(device)

##### training
epoch = 0
num_epochs = 100
while epoch < num_epochs:
    model.train()
    for s in source_trainloader:
        input_source, label_source, _ = s
        input_target, label_target = t
        optimizer.zero_grad()
​
        with torch.set_grad_enabled(True):
            # ### update using source data
            input_source = input_source.to(device)
            label_source = label_source.to(device)
​
            output_source = model(input_source)
            loss_source = objective(output_source['fc_source'], label_source)
​
            _, pred_source = torch.max(output_source['fc_source'], 1)
​
​
            ####### update using target data
            input_target = input_target.to(device)
            label_target = label_target.to(device)
​
            output_target = model(input_target)
            loss_target = objective(output_target['fc_target'], label_target)
​
            _, pred_target = torch.max(output_target['fc_target'], 1)
​
            loss_source.backward(retain_graph=True)
            loss_target.backward(retain_graph=True)
            optimizer.step()