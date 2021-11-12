"""UCI Mushroom dataset config.
"""

from .. import dataset
from .. import utils
from ..typing import CategoricalType

# flake8: noqa: E501
# pylint: disable=line-too-long

# Attribute Information:

# 1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises?: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment: attached=a,descending=d,free=f,notched=n
# 7. gill-spacing: close=c,crowded=w,distant=d
# 8. gill-size: broad=b,narrow=n
# 9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
# 10. stalk-shape: enlarging=e,tapering=t
# 11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 16. veil-type: partial=p,universal=u
# 17. veil-color: brown=n,orange=o,white=w,yellow=y
# 18. ring-number: none=n,one=o,two=t
# 19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

feature_dict = {
  "target":
    CategoricalType(["p", "e"]),
  "cap-shape":
    CategoricalType(["b", "c", "x", "f", "k", "s"]),
  "cap-surface":
    CategoricalType(["f", "g", "y", "s"]),
  "cap-color":
    CategoricalType(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"]),
  "bruises?":
    CategoricalType(["t", "f"]),
  "odor":
    CategoricalType(["a", "l", "c", "y", "f", "m", "n", "p", "s"]),
  "gill-attachment":
    CategoricalType(["a", "d", "f", "n"]),
  "gill-spacing":
    CategoricalType(["c", "w", "d"]),
  "gill-size":
    CategoricalType(["b", "n"]),
  "gill-color":
    CategoricalType([
      "k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"
    ]),
  "stalk-shape":
    CategoricalType(["e", "t"]),
  "stalk-root":
    CategoricalType(["b", "c", "u", "e", "z", "r", "?"]),
  "stalk-surface-above-ring":
    CategoricalType(["f", "y", "k", "s"]),
  "stalk-surface-below-ring":
    CategoricalType(["f", "y", "k", "s"]),
  "stalk-color-above-ring":
    CategoricalType(["n", "b", "c", "g", "o", "p", "e", "w", "y"]),
  "stalk-color-below-ring":
    CategoricalType(["n", "b", "c", "g", "o", "p", "e", "w", "y"]),
  "veil-type":
    CategoricalType(["p", "u"]),
  "veil-color":
    CategoricalType(["n", "o", "w", "y"]),
  "ring-number":
    CategoricalType(["n", "o", "t"]),
  "ring-type":
    CategoricalType(["c", "e", "f", "l", "n", "p", "s", "z"]),
  "spore-print-color":
    CategoricalType(["k", "n", "b", "h", "r", "o", "u", "w", "y"]),
  "population":
    CategoricalType(["a", "c", "n", "s", "v", "y"]),
  "habitat":
    CategoricalType(["g", "l", "m", "p", "u", "w", "d"]),
}
SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
      ),
      checksum="e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e",
      file_reader_args={
        "names": feature_dict,
        "dtype": feature_dict,
        "na_values": "?",
      }
    ),
}
