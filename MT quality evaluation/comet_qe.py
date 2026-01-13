from comet import load_from_checkpoint

model_path = r"D:\vscode\corpus data\comet_QE\checkpoints\model.ckpt"

# 强制从本地 hparams.yaml 重新加载
try:
    model = load_from_checkpoint(model_path, reload_hparams=True)
except TypeError:
    model = load_from_checkpoint(model_path, reload_params=True)

data = [
    {
        "src": "The output signal provides constant sync so the display never glitches.",
        "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
    },
    {
        "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
        "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
    },
    {
        "src": "I love China.",
        "mt": "我爱中国。"
    }
]

output = model.predict(data, batch_size=8, gpus=1)
print(output)