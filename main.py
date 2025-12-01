from sam3.model_builder import build_sam3_image_model

def main():
    checkpoint_path = "/Users/deekshith/Documents/Projects/vision-models/mlx_sam3/sam3-mod-weights/model.safetensors"
    build_sam3_image_model(
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
