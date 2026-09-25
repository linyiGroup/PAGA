import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.models import PAGAModel
from utils.utils import load_config, build_loader, train_one_epoch, evaluate, Logger


EXP = "unipep"
CONFIG_PATH = "./configs/config_%s.yaml" % EXP
CONFIG_PATH_MODEL = "./configs/config_model.yaml"

def main():
    config = load_config(CONFIG_PATH)
    config_model = load_config(CONFIG_PATH_MODEL)
    logger = Logger(config["output_path"]+ EXP +"/traing_log.txt")

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    train_loader = build_loader(config["data"]["train_csv"], config, True)
    val_loader = build_loader(config["data"]["val_csv"], config, False)


    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    model = PAGAModel(**config_model).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    epochs = config["training"]["epochs"]

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        
        msg = ""
        msg += f"EPOCH: {epoch} - Train loss: {train_loss:.6f}\n"
        msg += f"  Val loss: {val_metrics['loss']:.6f}\n"
        msg += f"   ROC-AUC \t PR-AUC \t PPVn \t Accuracy \t F1 \t MCC \n"
        msg += "   {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \t {: .2f} \n".format(
            val_metrics["roc_auc"] * 100,
            val_metrics["aupr"] * 100,
            val_metrics["ppvn"] * 100,
            val_metrics["accuracy"] * 100,
            val_metrics["f1"] * 100,
            val_metrics["mcc"] * 100,
        )

        logger.write(msg)
        torch.save(model.state_dict(), config["output_path"]+ EXP+"/PAGA%s.pt"%epoch)
            
        
        


if __name__ == "__main__":
    main()