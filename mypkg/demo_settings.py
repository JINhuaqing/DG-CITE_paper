from easydict import EasyDict as edict

simu_settings = edict()

simu_settings.setting1 = edict({
    "rho": 0, 
    "is_homo": True, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "norm",
    "cal_ratio": 0.25
})

simu_settings.setting2 = edict({
    "rho": 0.9, 
    "is_homo": True, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "norm",
    "cal_ratio": 0.25
})

simu_settings.setting3 = edict({
    "rho": 0.0, 
    "is_homo": False, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "norm",
    "cal_ratio": 0.25
})

simu_settings.setting4 = edict({
    "rho": 0.9, 
    "is_homo": False, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "norm",
    "cal_ratio": 0.25
})


simu_settings.setting5 = edict({
    "rho": 0, 
    "is_homo": True, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "t",
    "cal_ratio": 0.25
})

simu_settings.setting6 = edict({
    "rho": 0.9, 
    "is_homo": True, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "t",
    "cal_ratio": 0.25
})

simu_settings.setting7 = edict({
    "rho": 0.0, 
    "is_homo": False, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "t",
    "cal_ratio": 0.25
})

simu_settings.setting8 = edict({
    "rho": 0.9, 
    "is_homo": False, 
    "d": 10,
    "n": 3000, 
    "ntest": 1000,
    "err_type": "t",
    "cal_ratio": 0.25
})
