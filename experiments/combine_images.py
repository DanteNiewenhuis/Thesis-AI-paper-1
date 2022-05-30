# %%

from PIL import Image,  ImageDraw, ImageFont
import os
import pandas as pd

names = {"Branin": "Branin", "Easom": "Easom", "GoldsteinPrice": 'Goldstein-Price',
         "MartinGaddy": "Martin-Gaddy", "Mishra4": "Mishra4", "SixHump": "Six-HumpCamel"}

# names_space = {"Branin": "Branin", "Easom": "Easom", "GoldsteinPrice": 'Goldstein-<br>Price',
#          "MartinGaddy": "Martin-Gaddy", "Mishra4": "Mishra4", "SixHump": "Six-Hump<br>Camel"}

benchmarks = ["Branin", "Easom", "GoldsteinPrice",
              "MartinGaddy", "Mishra4", "SixHump"]


font = ImageFont.truetype("../assets/Balto-Medium.ttf", 100)
# %%
################################################################################
# Combining frames
################################################################################


def combine_frames(benchmark, selection, padding=10):
    images = []
    image_size = 1000

    for i in selection:
        images.append(Image.open(
            f"../results_paper_1/{benchmark}/frames-annotated/{i}.png"))

    extra_height = 0
    combined_image_size = (image_size*len(selection),
                           image_size + extra_height)
    text = names[benchmark]
    text_size = font.getsize(text)

    combined_image = Image.new("RGB", size=combined_image_size, color="white")

    for i, image in enumerate(images):
        combined_image.paste(image, (i*image_size, extra_height))

    d = ImageDraw.Draw(combined_image)

    xy = (20, 50)

    d.rectangle((xy[0] - padding, xy[1] - padding, xy[0] + text_size[0] + padding,
                xy[1] + text_size[1] + padding), fill=(200, 200, 200), outline=(10, 10, 10), width=5)

    d.text(xy, text, fill=(0, 0, 0), font=font)

    combined_image.save(f"../results_paper_1/{benchmark}/combined.png", "PNG")


combine_frames("Branin", [0, 1, 2, 4, 7])
combine_frames("Easom", [0, 3, 4, 10, 15])
combine_frames("GoldsteinPrice", [0, 3, 4, 10, 20])
combine_frames("MartinGaddy", [0, 4, 5, 14, 22])
combine_frames("Mishra4", [0, 3, 4, 10, 22])
combine_frames("SixHump", [0, 9, 19, 20, 27])

# %%

################################################################################
# Combining evolutions
################################################################################

image_width = 5000
image_height = 1000

combined_image_height = 6*image_height

combined_image = Image.new("RGB", size=(image_width, combined_image_height))

for i, benchmark in enumerate(benchmarks):
    image = Image.open(
        f"../results_paper_1/{benchmark}/combined.png")

    combined_image.paste(image, (0, i*image_height))

# new_width = 1000

# combined_image = combined_image.resize((new_width, int(new_width*(combined_image_height / image_width))))
combined_image.save(f"../results_paper_1/combined_evolutions.png", "PNG")

# %%

################################################################################
# Combine four evolutions
################################################################################

image_width = 5000
image_height = 1000

combined_image_height = 4*image_height

combined_image = Image.new("RGB", size=(image_width, combined_image_height))

four_benchmarks = ["Easom", "GoldsteinPrice", "Mishra4", "SixHump"]
for i, benchmark in enumerate(four_benchmarks):
    image = Image.open(
        f"../results_paper_1/{benchmark}/combined.png")

    combined_image.paste(image, (0, i*image_height))

# new_width = 1000
# combined_image = combined_image.resize((new_width, int(new_width*(combined_image_height / image_width))))

combined_image.save(f"../results_paper_1/combined_evolutions_four.png", "PNG")


# %%

################################################################################
# Combining bases
################################################################################


images = []
height = 1000
width = 1000

for b in ["Branin", "Easom", "GoldsteinPrice", "MartinGaddy", "Mishra4", "SixHump"]:
    images.append(Image.open(f"../results_paper_1/base_benchmarks/{b}.png"))


combined_image = Image.new("RGB", size=(width*3, height*2))

for i, image in enumerate(images):
    col = i % 3
    row = int(i / 3)

    print(f"{i=}, {col=}, {row=}")

    combined_image.paste(image, (col*width, row*height))

# combined_image = combined_image.resize((1200, 800))

combined_image.save(
    f"../results_paper_1/base_benchmarks/combined3000.png", "PNG")

# %%
################################################################################
# Annotate
################################################################################


def annotate_frames(benchmark: str):

    df = pd.read_csv(f"../results_paper_1/{benchmark}/run-unique.csv")
    num_models = len(df)
    fnt_anno = ImageFont.truetype("../assets/Balto-Light.ttf", 60)
    padding = 5

    for i in range(num_models):
        image = Image.open(
            f"../results_paper_1/{benchmark}/frames/{i}.png")

        model_text = f"{i+1}/{num_models}"
        MOD_text = f"{df.iloc[i]['val']:.5f}"

        model_text_size = fnt_anno.getsize(model_text)
        MOD_text_size = fnt_anno.getsize(MOD_text)

        model_position = (150, 900)
        MOD_position = (850, 900)

        draw = ImageDraw.Draw(image)

        # Add Model
        xy = (model_position[0], model_position[1])
        draw.rectangle((xy[0] - padding, xy[1] - padding, xy[0] + model_text_size[0] + padding,
                        xy[1] + model_text_size[1] + padding), fill=(240, 240, 240), outline=(100, 100, 100), width=2)
        draw.text(xy, model_text, fill=(0, 0, 0), font=fnt_anno)

        # Add MOD
        xy = (MOD_position[0] - MOD_text_size[0], MOD_position[1])
        draw.rectangle((xy[0] - padding, xy[1] - padding, xy[0] + MOD_text_size[0] + padding,
                        xy[1] + MOD_text_size[1] + padding), fill=(240, 240, 240), outline=(100, 100, 100), width=2)
        draw.text(xy, MOD_text, fill=(0, 0, 0), font=fnt_anno)

        if not os.path.exists(f"../results_paper_1/{benchmark}/frames-annotated"):
            os.mkdir(f"../results_paper_1/{benchmark}/frames-annotated")
        image.save(
            f"../results_paper_1/{benchmark}/frames-annotated/{i}.png", "PNG")


for b in ["Branin", "Easom", "GoldsteinPrice", "MartinGaddy", "Mishra4", "SixHump"]:
    annotate_frames(b)
# %%
