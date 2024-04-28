'''
Date: 2024-04-23 02:19:26
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-27 21:51:45
FilePath: /SparseTeMPO/unitest/test_power_fit.py
'''
"""
Date: 2024-04-22 12:08:46
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-22 17:54:07
FilePath: /SparseTeMPO/unitest/test_crosstalk_fit.py
"""
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import torch

class PowerMLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden = 32
        self.exp = torch.nn.Parameter(torch.tensor(-0.1))
        self.features = torch.nn.Sequential(
            torch.nn.Linear(2, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1),
        )
    def forward(self, x):
        # x = torch.stack([x[:, 0], torch.exp(self.exp * x[:, 1])], -1)
        return self.features(x).abs() * (1-torch.exp(self.exp * x[:, 1:2]))
    
def fit_power():
    df = pd.read_csv("./unitest/MZIPower.csv")
    print(df)
    # end = 22
    end = -5 # 7um at P_pi
    power = df.iloc[2:end, 1].astype('float32').to_numpy()
    delta_phases = []
    distances = []
    for i in range(1, 13):
        delta_phi = df.iloc[2:end, 1+4*i].astype('float32').to_numpy()-(df.iloc[2:end, 4*i].astype('float32').to_numpy())
        distance = df.iloc[0:1, i*4-2].astype('float32').repeat(len(delta_phi)).to_numpy()
        delta_phases.append(delta_phi)
        distances.append(distance)
    # print(power)
    # print(delta_phases)
    ideal_distance = distances.pop(-1)
    ideal_delta_phases = delta_phases.pop(-1)
    # ideal_distance = distances[-1]
    # ideal_delta_phases = delta_phases[-1]
    ## delta phase cannot be higher then ideal case
    ## larger distance must have larger delta phase
    for i in range(1, len(delta_phases)):
        delta_phases[i] = np.minimum(np.maximum(delta_phases[i-1], delta_phases[i]), ideal_delta_phases)
    X_data = np.stack([np.concatenate(delta_phases, 0), np.concatenate(distances, 0)], -1)
    # print(X_data)
    Y_data = np.concatenate([power]*len(delta_phases), 0)
    # print(Y_data)

    # [1, a, b, a^2, ab, b^2, a^3, a^2b, ab^2, b^3]
    poly = PolynomialFeatures(degree=2)
    print(list(poly._combinations(
        2, 1, 3, interaction_only=False, include_bias=True
    )))
    X_poly = poly.fit_transform(X_data)
    X_feat = X_poly
    # X_feat = np.stack([
    #     X_poly[:, 1], # phi
    #     X_poly[:, 3], # phi^2
    #     X_poly[:, 4], # phi * dist 
    #     # X_poly[:, 6], # phi^3
    #     X_poly[:, 7], # phi^2 * dist
    #     # X_poly[:, 8], # phi * dist^2
    # ], -1)
    # Linear Regression model
    model = LinearRegression(fit_intercept=True)
    reg = model.fit(X_feat, Y_data)
    print(reg.score(X_feat[:18], Y_data[:18]))
    print(reg.coef_)
    print(reg.intercept_)
    # X = []
    # for dist in np.linspace(7, 25, 30):
    #     for delta_phi in np.linspace(0, np.pi/2, 30):
    #         X.append([delta_phi, dist])
    # X = np.array(X)
    # X_poly = poly.fit_transform(X)
    # Y_pred = model.predict(X_poly)


    Y_pred = model.predict(X_feat)




    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_data[:, 0], X_data[:,1], Y_data, c="b", label="data", marker="^")

    # ax.scatter(X[:, 0], X[:,1], Y_pred, c="r", label="data", marker=".",)
    ax.scatter(X_data[:, 0], X_data[:,1], Y_pred, c="r", label="data", marker=".",)
    ax.set_xlabel("delta_phi (rad.)")
    ax.set_ylabel("distance (um)")
    ax.set_zlabel("power (mW)")
    plt.savefig("./unitest/figs/power_fit_3d.png", dpi=300)

    fig, axes = plt.subplots(1,len(delta_phases), figsize=(3*len(delta_phases), 4))
    X_data = X_data.reshape([len(delta_phases), -1, 2])
    Y_data = Y_data.reshape([len(delta_phases), -1])
    Y_pred = Y_pred.reshape([len(delta_phases), -1])
    for i, (x, y, y_pred) in enumerate(zip(X_data, Y_data, Y_pred)):
        axes[i].scatter(x[:, 0],  y, c="b", label="simulated", marker="^")
        axes[i].scatter(x[:, 0],  y_pred, c="r", label="predicted", marker=".")

        axes[i].set_xlabel("delta_phi (rad.)")
        axes[i].set_ylabel("power (mW)")
        axes[i].set_title(f"distance: {distances[i][0]} um")
        axes[i].set_xlim([0, np.pi/2+0.5])
        axes[i].set_ylim([0, 7.5])
        # axes[i].set_xlim([0, np.pi/2+2.5])
        # axes[i].set_ylim([0, 9.5])
        axes[i].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("./unitest/figs/power_fit.png", dpi=300)

  
def fit_power_mlp():
    df = pd.read_csv("./unitest/MZIPower.csv")
    print(df)
    # end = 22
    end = -5 # 7um at P_pi
    power = df.iloc[2:end, 1].astype('float32').to_numpy()
    delta_phases = []
    distances = []
    for i in range(1, 13):
        delta_phi = df.iloc[2:end, 1+4*i].astype('float32').to_numpy()-(df.iloc[2:end, 4*i].astype('float32').to_numpy())
        distance = df.iloc[0:1, i*4-2].astype('float32').repeat(len(delta_phi)).to_numpy()
        delta_phases.append(delta_phi)
        distances.append(distance)
    # print(power)
    # print(delta_phases)
    ideal_distance = distances.pop(-1)
    ideal_delta_phases = delta_phases.pop(-1)
    # ideal_distance = distances[-1]
    # ideal_delta_phases = delta_phases[-1]
    ## delta phase cannot be higher then ideal case
    ## larger distance must have larger delta phase
    for i in range(1, len(delta_phases)):
        delta_phases[i] = np.minimum(np.maximum(delta_phases[i-1], delta_phases[i]), ideal_delta_phases)
    X_data = np.stack([np.concatenate(delta_phases, 0), np.concatenate(distances, 0)], -1)
    # print(X_data)
    Y_data = np.concatenate([power]*len(delta_phases), 0)
    # print(Y_data)

    device = "cuda:0"
    # [1, a, b, a^2, ab, b^2, a^3, a^2b, ab^2, b^3]
    model = PowerMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-5)
    criterion = torch.nn.MSELoss()


    poly = PolynomialFeatures(degree=2)
    print(list(poly._combinations(
        2, 1, 3, interaction_only=False, include_bias=True
    )))
    # X_poly = poly.fit_transform(X_data)
    # X_feat = X_poly
    X_feat = X_data
    # X_feat = np.stack([
    #     X_poly[:, 1], # phi
    #     X_poly[:, 3], # phi^2
    #     X_poly[:, 4], # phi * dist 
    #     # X_poly[:, 6], # phi^3
    #     X_poly[:, 7], # phi^2 * dist
    #     # X_poly[:, 8], # phi * dist^2
    # ], -1)
    # Linear Regression model
    X_feat_cuda = torch.from_numpy(X_feat).to(device)
    Y_data_cuda = torch.from_numpy(Y_data).to(device)[..., None]
    print(X_feat_cuda.shape)
    print(Y_data_cuda.shape)
    for i in range(1500):
        optimizer.zero_grad()
        Y_pred = model(X_feat_cuda)
   
        loss = criterion(Y_pred, Y_data_cuda)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print(loss.item(), Y_pred[2].data, Y_data_cuda[2])
    Y_pred = Y_pred.cpu().detach().numpy()
    print(r2_score(Y_pred[:18], Y_data[:18]))
    # exit(0)
    # Y_pred *= 10
    # Y_data *= 10


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_data[:, 0], X_data[:,1], Y_data, c="b", label="data", marker="^")

    # ax.scatter(X[:, 0], X[:,1], Y_pred, c="r", label="data", marker=".",)
    ax.scatter(X_data[:, 0], X_data[:,1], Y_pred, c="r", label="data", marker=".",)
    ax.set_xlabel("delta_phi (rad.)")
    ax.set_ylabel("distance (um)")
    ax.set_zlabel("power (mW)")
    plt.savefig("./unitest/figs/power_fit_mlp_3d.png", dpi=300)

    fig, axes = plt.subplots(1,len(delta_phases), figsize=(3*len(delta_phases), 4))
    X_data = X_data.reshape([len(delta_phases), -1, 2])
    Y_data = Y_data.reshape([len(delta_phases), -1])
    Y_pred = Y_pred.reshape([len(delta_phases), -1])
    for i, (x, y, y_pred) in enumerate(zip(X_data, Y_data, Y_pred)):
        axes[i].scatter(x[:, 0],  y, c="b", label="simulated", marker="^")
        axes[i].scatter(x[:, 0],  y_pred, c="r", label="predicted", marker=".")

        axes[i].set_xlabel("delta_phi (rad.)")
        axes[i].set_ylabel("power (mW)")
        axes[i].set_title(f"distance: {distances[i][0]} um")
        axes[i].set_xlim([0, np.pi/2+0.5])
        axes[i].set_ylim([0, 7.5])
        # axes[i].set_xlim([0, np.pi/2+2.5])
        # axes[i].set_ylim([0, 9.5])
        axes[i].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("./unitest/figs/power_fit_mlp.png", dpi=300)


def fit_power_interp():
    df = pd.read_csv("./unitest/MZIPower.csv")
    print(df)
    # end = 22
    end = -5 # 7um at P_pi
    power = df.iloc[2:end, 1].astype('float32').to_numpy()
    delta_phases = []
    distances = []
    for i in range(1, 13):
        delta_phi = df.iloc[2:end, 1+4*i].astype('float32').to_numpy()-(df.iloc[2:end, 4*i].astype('float32').to_numpy())
        distance = df.iloc[0:1, i*4-2].astype('float32').repeat(len(delta_phi)).to_numpy()
        delta_phases.append(delta_phi)
        distances.append(distance)
    # print(power)
    # print(delta_phases)
    ideal_distance = distances.pop(-1)
    ideal_delta_phases = delta_phases.pop(-1)
    # ideal_distance = distances[-1]
    # ideal_delta_phases = delta_phases[-1]
    ## delta phase cannot be higher then ideal case
    ## larger distance must have larger delta phase
    for i in range(1, len(delta_phases)):
        delta_phases[i] = np.minimum(np.maximum(delta_phases[i-1], delta_phases[i]), ideal_delta_phases)
    X_data = np.stack([np.concatenate(delta_phases, 0), np.concatenate(distances, 0)], -1)
    # print(X_data)
    Y_data = np.concatenate([power]*len(delta_phases), 0)
    # print(Y_data)

    interp = LinearNDInterpolator(X_data, Y_data)

    fig, axes = plt.subplots(1,len(delta_phases), figsize=(3*len(delta_phases), 4))
    X_data = X_data.reshape([len(delta_phases), -1, 2])
    Y_data = Y_data.reshape([len(delta_phases), -1])
    xs = np.linspace(0, 2, 100)
    for i, (x, y) in enumerate(zip(X_data, Y_data)):
        axes[i].scatter(x[:, 0],  y, c="b", label="simulated", marker="o")
        distance = np.zeros_like(xs) + distances[i][0]
        x = np.stack([xs, distance], -1)
        y_pred = interp(x)
        axes[i].plot(xs,  y_pred, c="r", label="predicted")

        axes[i].set_xlabel("delta_phi (rad.)")
        axes[i].set_ylabel("power (mW)")
        axes[i].set_title(f"distance: {distances[i][0]} um")
        axes[i].set_xlim([0, np.pi/2+0.5])
        axes[i].set_ylim([0, 7.5])
        # axes[i].set_xlim([0, np.pi/2+2.5])
        # axes[i].set_ylim([0, 9.5])
        axes[i].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("./unitest/figs/power_interp.png", dpi=300)

if __name__ == "__main__":
    # fit_power()
    # fit_power_mlp()
    fit_power_interp()