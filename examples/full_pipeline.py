import py3DCal as p3d
from py3DCal import datasets, models

if __name__ == "__main__":
    # Data Collection
    digit = p3d.DIGIT("D20966")
    ender3 = p3d.Ender3("/dev/ttyUSB0")
    calibrator = p3d.Calibrator(printer=ender3, sensor=digit)

    calibrator.probe(calibration_file_path="misc/probe_points.csv")

    # Data Annotation
    p3d.annotate(dataset_path="./sensor_calibration_data", probe_radius_mm=2.0)

    # Model Training
    my_dataset = datasets.TactileSensorDataset(root='./sensor_calibration_data')
    touchnet = models.TouchNet()

    p3d.train_model(model=touchnet, dataset=my_dataset, num_epochs=60, batch_size=64)

    # Depthmap Generation
    depthmap = p3d.get_depthmap(model=touchnet, image_path="path/to/target/image", blank_image_path="./sensor_calibration_data/blank_images/blank.png")