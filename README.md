# Intelligent-Scissors
The Intelligent Scissors can be used to select an image region defined by strong color-changes at the edges.

You can learn more about it in papers:
- [Intelligent Scissors for Image Composition](https://courses.cs.washington.edu/courses/cse455/02wi/readings/mort-sigg95.pdf)
- [Interactive Segmentation with Intelligent Scissors](https://pdf.sciencedirectassets.com/272316/1-s2.0-S1077316900X00068/1-s2.0-S1077316998904804/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBgaCXVzLWVhc3QtMSJHMEUCIAWCjFZL7PQFDvUmwMnc5US2%2FVONJ%2F35D8YSbksXWzmRAiEAjcUTMNaKtHM9Fyl7mnXrZyI%2F3cN%2BZ9dnjh9wLft7vzkqtAMIQRADGgwwNTkwMDM1NDY4NjUiDK3za45FaD20672xiCqRAzmaa28z8IzGVOPRTwsJ%2F8hbivchKVoxzXmSqzyEGgUFrLoAimHRtHnIPIuUQvTEl%2FE%2Boh%2FFmFWEUDxTsUu7ttIUX1VFrwQ03Jhcydpj20g5o%2FEsuYpo%2FCIyWq0FmJQMruOn%2BvLeQesm6ODeV%2Bq5jgZ3hPGal3t7QqfaAc8EQtgm2pj2NHtLFBBazJIulPgmKegrwjhTilWK%2Bq8zfEfM8DI8fQb%2BRexIJ3gr6EpIFtDA40SRx2jj7MX%2B3cAErSRHnn%2B2R3F%2FklE%2FwfCs3IJT3Ff%2BOgGMPNc2Smx5OdwdIPCWJnuPDD5AtqViVDK8JxwPLsUv6EQPkecuDz0f5rGl3KuRL2orR6LI6LC7uoNMT9rP1si8joGts9C0zU9vjTT1po9%2ByIxQjMU1An5QZD8VhTpZJ8qUaFfzF6yc%2F9hhybKdjC6ZCEmNBFSV48KSO14y0e4rOeEYWokbFy3759ULPb53xkoJa1NBzY7hJb4PotAtWUUhTyE%2BWuShRpEsCr7EJlgkd%2BBQMlstNeFwMy7OVo1IMKHk%2F%2FQFOusBLkihikZdSCk%2FpXCngTV%2BurqIPdtnBExW4Y%2FpVrUVk89D11TzMy368ugka1A2aLEJr02F1kEsyKNarhC3%2FuDcaJU%2BJLblbbOElrjs8IpOiGT976Iq90oXw6VM%2BNvJZsE53sTBcRIrV%2FSsQXcJzLqUr1KpJtSySTW8D9GmeHN7KDxjMwLDshJuE8J7EK5FTFahnOhhaCA34YWaC8PE3uZu%2BqJj0GRwaOOCNMudg9xAPcPklrmnfPTs6jNdeHyM0mcAefvVO%2BpWmoQRYJHGcGvm0CqYpWRgbTDMvDMXt8XR5JDZtKHAV%2FG9OpC6Zw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200422T074335Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7OHFSYOL%2F20200422%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0f94e5c520d1726100ee62b9d40dc924c5fabc87c0cd1d88853e9a0e7a8e0267&hash=39309b5cca6bffdbc7dee114145cacc84838a91f8349d98390c5452a8e69a347&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1077316998904804&tid=spdf-bf7d7bea-c627-4dce-bec0-272fa3d00d20&sid=ee51f81c2575484c068ab8b9d830a36c9d73gxrqb&type=client)

<p align="center">
  <img width="690" height="413" src="https://i.ibb.co/6XxjNSp/kakdu.png">
</p>

## Installation
`pip install intelligent-scissors`


## Usage

To use in your program
```python

from scissors.feature_extraction import Scissors

image = ...
scissors = Scissors(image)

seed_x, seed_y = ...
free_x, free_y = ...
path = scissors.find_path(seed_x, seed_y, free_x, free_y)
```

Also you can run a simple demo

```python
from scissors.gui import run_demo

file_name = 'image.png'
run_demo(file_name)
```

## Details

The current implementation includes 

* Cursor snap
* Static features
* Dynamic features
* On-the-fly Training
* Unrestricted graph search

On-the-fly Training allows you to select a “good” initial-boundary segment. 
However, this results in poor performance for new segments with different intensity/gradient magnitudes.
To overcome this, you should first try to select a small region of the new segment to create correct dynamic features. 
