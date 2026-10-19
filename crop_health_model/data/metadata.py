spectrometry_cassava_dataset = {
    "ids": [
        ("7101239", "Field Experiment.part1.rar"),
        ("7101240", "Field Experiment.part2.rar"),
        ("7101238", "Field Experiment.part3.rar"),
        ("7101244", "ScreenHouse Experiment.part1.rar"),
        ("7101243", "ScreenHouse Experiment.part2.rar"),
        ("7101245", "ScreenHouse Experiment.part3.rar"),
    ],
    "folder": "spectrometry-cassava-dataset",
    "crop_type": "cassava",
    "classes": [
        {
            "raw": "CBSD",
            "clean": "CBSD",
            "full": "Cassava Brown Streak Disease",
            "count": -1,
        },
        {"raw": "CMD", "clean": "CMD", "full": "Cassava Mosaic Disease", "count": -1},
        {"raw": "HLT", "clean": "HLT", "full": "Healthy", "count": -1},
    ],
    "total_image_count": 7860,
}

cassava_dataset_uganda = {
    "ids": [
        ("6419132", "cbsd-2.rar"),
        ("6419150", "cmd_001.rar"),
        ("6419127", "cbsd-1.rar"),
        ("6419152", "cmd_002.rar"),
        ("6419151", "cmd_003.rar"),
        ("6420454", "cmd_004.rar"),
        ("6420455", "cmd_005.rar"),
        ("6419125", "healthy_001.rar"),
        ("6419126", "healthy_002.rar"),
    ],
    "folder": "cassava-dataset-uganda",
    "crop_type": "cassava",
    "classes": [
        {
            "raw": "cbsd",
            "clean": "CBSD",
            "full": "Cassava Brown Streak Disease",
            "count": 5000,
        },
        {"raw": "cmd", "clean": "CMD", "full": "Cassava Mosaic Disease", "count": 5000},
        {"raw": "healthy", "clean": "HLT", "full": "Healthy", "count": 5000},
    ],
    "total_image_count": 15000,
}

maize_dataset_tanzania = {
    "ids": [
        ("6966997", "HEATHLY.zip"),
        ("6962066", "MLN.zip"),
        ("6966998", "MSV_1.zip"),
        ("6962930", "MSV_2.zip"),
    ],
    "folder": "maize-dataset-tanzania",
    "crop_type": "maize",
    "classes": [
        {"raw": "MLN", "clean": "MLN", "full": "Maize Lethal Necrosis", "count": 5068},
        {"raw": "MSV", "clean": "MSV", "full": "Maize Streak Virus", "count": 6667},
        {"raw": "HEATHLY", "clean": "HLT", "full": "Healthy", "count": 5542},
    ],
    "total_image_count": 17277,
}

maize_dataset_namibia = {
    "ids": [("6367418", "FAW.rar"), ("6367417", "healthy.rar"), ("6367413", "MSV.rar")],
    "folder": "maize-dataset-namibia",
    "crop_type": "maize",
    "classes": [
        {"raw": "FAW", "clean": "FAW", "full": "Fall Armyworm", "count": 3414},
        {"raw": "MSV", "clean": "MSV", "full": "Maize Streak Virus", "count": 3009},
        {"raw": "healthy", "clean": "HLT", "full": "Healthy", "count": 2611},
    ],
    "total_image_count": 9034,
}

maize_dataset_uganda = {
    "ids": [
        ("6045938", "Healthy_1.rar"),
        ("6045939", "Healthy_2.rar"),
        ("6284851", "MLB_1.rar"),
        ("6284853", "MLB_2.rar"),
        ("6299680", "MSV_1.rar"),
        ("6299689", "MSV_2.rar"),
    ],
    "folder": "maize-dataset-uganda",
    "crop_type": "maize",
    "classes": [
        {"raw": "MLB", "clean": "MLB", "full": "Maize Leaf Blight", "count": 5279},
        {"raw": "MSV", "clean": "MSV", "full": "Maize Streak Virus", "count": 5216},
        {"raw": "Healthy", "clean": "HLT", "full": "Healthy", "count": 5326},
    ],
    "total_image_count": 15821,
}

beans_dataset_uganda = {
    "ids": [
        ("5857998", "als_1.zip"),
        ("5858001", "als_2.zip"),
        ("5857999", "bean_rust_1.zip"),
        ("5858000", "bean_rust_2.zip"),
        ("5741405", "healthy_1.zip"),
        ("5741402", "healthy_2.zip"),
    ],
    "folder": "beans-dataset-uganda",
    "crop_type": "beans",
    "classes": [
        {"raw": "als", "clean": "ALS", "full": "Angular Leaf Spot", "count": 5031},
        {"raw": "bean_rust", "clean": "BR", "full": "Bean Rust", "count": 5020},
        {"raw": "healthy", "clean": "HLT", "full": "Healthy", "count": 5284},
    ],
    "total_image_count": 15335,
}

bananas_dataset_tanzania = {
    "ids": [
        ("7071415", "BLACK SIGATOKA_1.zip"),
        ("6078711", "BLACK SIGATOKA_2.zip"),
        ("7071416", "FUSARIUM WILT_1.zip"),
        ("7077487", "FUSARIUM WILT_2.zip"),
        ("7077478", "HEALTHY_1.zip"),
        ("7077403", "HEALTHY_2.zip"),
    ],
    "folder": "bananas-dataset-tanzania",
    "crop_type": "bananas",
    "classes": [
        {
            "raw": "BLACK SIGATOKA",
            "clean": "BS",
            "full": "Black Sigatoka",
            "count": 6147,
        },
        {
            "raw": "FUSARIUM WILT",
            "clean": "FW",
            "full": "Fusarium Wilt Race 1",
            "count": 5038,
        },
        {"raw": "HEALTHY", "clean": "HLT", "full": "Healthy", "count": 5883},
    ],
    "total_image_count": 17068,
}

cocoa_dataset_karagro = {
    "ids": [
        ("6381346", "anthracnose_01.zip"),
        ("6381376", "anthracnose_02.zip"),
        ("6381402", "anthracnose_03.zip"),
        ("6381403", "anthracnose_04.zip"),
        ("6390000", "cssvd_01.zip"),
        ("6390066", "cssvd_02.zip"),
        ("6390067", "cssvd_03.zip"),
        ("6412512", "cssvd_04.rar"),
        ("6390068", "healthy_01.zip"),
        ("6390069", "healthy_02.zip"),
        ("6390070", "healthy_03.zip"),
        ("6392756", "healthy_04.rar"),
        ("6412511", "healthy_05.rar"),
    ],
    "folder": "cocoa-dataset-karagro",
    "crop_type": "cocoa",
    "classes": [
        {"raw": "anthracnose", "clean": "ANT", "full": "Anthracnose", "count": 5133},
        {
            "raw": "cssvd",
            "clean": "CSSVD",
            "full": "Cocoa Swollen Shoot Virus Disease",
            "count": 6757,
        },
        {"raw": "healthy", "clean": "HLT", "full": "Healthy", "count": 5056},
    ],
    "total_image_count": 16946,
}

maize_dataset_karagro = {
    "ids": [
        ("6418125", "faw_01.zip"),
        ("6418150", "faw_02.zip"),
        ("6418153", "faw_03.rar"),
        ("6418156", "faw_04.zip"),
        ("6418158", "faw_05.rar"),
        ("6418190", "faw_06.rar"),
        ("6418191", "faw_07.rar"),
        ("6417970", "healthy_01.rar"),
        ("6417972", "healthy_02.rar"),
        ("6417980", "healthy_03.rar"),
        ("6417981", "healthy_04.rar"),
        ("6417982", "healthy_05.rar"),
        ("6418018", "healthy_06.rar"),
        ("6418084", "healthy_07.rar"),
        ("6417320", "maize_streak_01.rar"),
        ("6417321", "maize_streak_02.rar"),
        ("6417334", "maize_streak_03.rar"),
        ("6417960", "maize_streak_04.rar"),
        ("6417967", "maize_streak_05.rar"),
        ("6417969", "maize_streak_06.rar"),
    ],
    "folder": "maize-dataset-karagro",
    "crop_type": "maize",
    "classes": [
        {
            "raw": "maize_streak",
            "clean": "MSV",
            "full": "Maize Streak Virus",
            "count": 5063,
        },
        {"raw": "faw", "clean": "FAW", "full": "Fall Armyworm", "count": 5110},
        {"raw": "healthy", "clean": "HLT", "full": "Healthy", "count": 5392},
    ],
    "total_image_count": 16946,
}

all_datasets = [
    spectrometry_cassava_dataset,
    cassava_dataset_uganda,
    maize_dataset_tanzania,
    maize_dataset_namibia,
    maize_dataset_uganda,
    beans_dataset_uganda,
    maize_dataset_karagro,
]
