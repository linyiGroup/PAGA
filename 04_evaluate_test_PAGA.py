import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.models import PAGAModel
from utils.utils import load_config, build_loader, train_one_epoch, evaluate, Logger


EXP = "reftcr"
CONFIG_PATH = "./configs/config_%s.yaml" % EXP
CONFIG_PATH_MODEL = "./configs/config_model.yaml"


def main():
    config = load_config(CONFIG_PATH)
    config_model = load_config(CONFIG_PATH_MODEL)
    logger = Logger(config["output_path"]+ EXP +"/test_log.txt")

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    val_loader = build_loader(config["data"]["val_csv"], config, False)
    independent_test_loader = build_loader(config["data"]["independent_test_csv"], config, False)
    external_test_loader = build_loader(config["data"]["external_test_csv"], config, False)

    print("Validation samples:", len(val_loader.dataset))
    print("Independent test samples:", len(independent_test_loader.dataset))
    print("External test samples:", len(external_test_loader.dataset))

    model = PAGAModel(**config_model).to(device)
    model.load_state_dict(torch.load(config["testing"]["model_path"]))

    criterion = torch.nn.CrossEntropyLoss()

    val_metrics = evaluate(model, val_loader, criterion, device)

    independent_test_metrics = evaluate(
        model, independent_test_loader, criterion, device
    )
    external_test_metrics = evaluate(
        model, external_test_loader, criterion, device
    )

    msg = ""

    msg += f"Validation loss: {val_metrics['loss']:.6f}\n"
    msg += f"   ROC-AUC \t PR-AUC \t PPVn \t Accuracy \t F1 \t MCC \n"
    msg += "   {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \n".format(
        val_metrics["roc_auc"] * 100,
        val_metrics["aupr"] * 100,
        val_metrics["ppvn"] * 100,
        val_metrics["accuracy"] * 100,
        val_metrics["f1"] * 100,
        val_metrics["mcc"] * 100,
    )

    msg += f"  independent test loss: {independent_test_metrics['loss']:.6f}\n"
    msg += f"   ROC-AUC \t PR-AUC \t PPVn \t Accuracy \t F1 \t MCC \n"
    msg += "   {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \n".format(
        independent_test_metrics["roc_auc"] * 100,
        independent_test_metrics["aupr"] * 100,
        independent_test_metrics["ppvn"] * 100,
        independent_test_metrics["accuracy"] * 100,
        independent_test_metrics["f1"] * 100,
        independent_test_metrics["mcc"] * 100,
    )
    msg += f"  external test loss: {external_test_metrics['loss']:.6f}\n"
    msg += f"   ROC-AUC \t PR-AUC \t PPVn \t Accuracy \t F1 \t MCC \n"
    msg += "   {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \n".format(
        external_test_metrics["roc_auc"] * 100,
        external_test_metrics["aupr"] * 100,
        external_test_metrics["ppvn"] * 100,
        external_test_metrics["accuracy"] * 100,
        external_test_metrics["f1"] * 100,
        external_test_metrics["mcc"] * 100,
    )

        
    logger.write(msg)
        


if __name__ == "__main__":
        main()