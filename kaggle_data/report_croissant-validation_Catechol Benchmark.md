# CROISSANT VALIDATION REPORT
================================================================================
## VALIDATION RESULTS
--------------------------------------------------------------------------------
Starting validation for file: catechol-benchmark-metadata.json
### JSON Format Validation
✓
The file is valid JSON.
### Croissant Schema Validation
✓
The dataset passes Croissant validation.
### Records Generation Test
✓
Record set 'acs_pca_descriptors_lookup.csv' passed validation.
Record set 'catechol_full_data_yields.csv' passed validation.
Record set 'catechol_single_solvent_yields.csv' passed validation.
Record set 'claisen_data_clean.csv' passed validation.
Record set 'smiles_lookup.csv' passed validation.
Record set 'spange_descriptors_lookup.csv' passed validation.
## JSON-LD REFERENCE
================================================================================
```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {
      "@id": "cr:data",
      "@type": "@json"
    },
    "dataBiases": "cr:dataBiases",
    "dataCollection": "cr:dataCollection",
    "dataType": {
      "@id": "cr:dataType",
      "@type": "@vocab"
    },
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isEnumeration": "cr:isEnumeration",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "personalSensitiveInformation": "cr:personalSensitiveInformation",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "wd": "https://www.wikidata.org/wiki/",
    "@base": "cr_base_iri/"
  },
  "alternateName": "Time-series Solvent Selection Data for Few-shot Machine Learning",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "license": {
    "@type": "sc:CreativeWork",
    "name": "MIT",
    "url": "https://www.mit.edu/~amini/LICENSE.md"
  },
  "distribution": [
    {
      "contentUrl": "https://www.kaggle.com/api/v1/datasets/download/aichemy/catechol-benchmark?datasetVersionNumber=4",
      "contentSize": "98.305 KB",
      "md5": "zW6hk+us6JWXnT7xutqIUA==",
      "encodingFormat": "application/zip",
      "@id": "archive.zip",
      "@type": "cr:FileObject",
      "name": "archive.zip",
      "description": "Archive containing all the contents of the Catechol Benchmark dataset"
    },
    {
      "contentUrl": "acs_pca_descriptors_lookup.csv",
      "containedIn": {
        "@id": "archive.zip"
      },
      "encodingFormat": "text/csv",
      "@id": "acs_pca_descriptors_lookup.csv_fileobject",
      "@type": "cr:FileObject",
      "name": "acs_pca_descriptors_lookup.csv",
      "description": "Translation of solvent names to the ML-readable representation: ACS Solvent Selection Guide's principle component analysis representation (ACS PCA)"
    },
    {
      "contentUrl": "catechol_full_data_yields.csv",
      "containedIn": {
        "@id": "archive.zip"
      },
      "encodingFormat": "text/csv",
      "@id": "catechol_full_data_yields.csv_fileobject",
      "@type": "cr:FileObject",
      "name": "catechol_full_data_yields.csv",
      "description": "Full data set with mixture solvents.\n\n| Name | Type | Description |\n|--------|--------|--------|\n| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|\n| `SOLVENT {A/B} NAME` | str | Chemical name of the solvents; used as a key when looking up featurizations|\n| `SolventB%` | float | Percent concentration of solvent B; the rest of the solvent is made up of solvent A|\n| `Residence Time` | float | Time (in minutes) of the reaction|\n| `Temperature`| float | Temperature (in Celsius) of the reaction|\n| `SM` | float | Quantity of starting material measured (yield %)|\n| `Product {2/3}` | float | Quantity of product {2/3} measured (yield %)| \n| `SM SMILES` | str | SMILES string representation of starting material|\n| `Product {2/3} SMILES` | str | SMILES string representation of product 2/3|\n| `SOLVENT {A/B} SMILES` | str | SMILES string representation of solvent {A/B}|\n| `SOLVENT {A/B} Ratio` | str |  Ratio of the component solvents in solvent {A/B} |\n| `Reaction SMILES {A/B}` | str | SMILES string representation of reaction between starting material and Solvent {A/B}  |\n| `RAMP NUM` | str | Ramp number |\n\n\n\n**Inputs:** Residence Time, Temperature, SOLVENT A NAME, SOLVENT B NAME, SolventB%\n**Outputs:** SM, Product 2, Product 3"
    },
    {
      "contentUrl": "catechol_single_solvent_yields.csv",
      "containedIn": {
        "@id": "archive.zip"
      },
      "encodingFormat": "text/csv",
      "@id": "catechol_single_solvent_yields.csv_fileobject",
      "@type": "cr:FileObject",
      "name": "catechol_single_solvent_yields.csv",
      "description": "Single-solvent dataset. \n\n| Name | Type | Description |\n|--------|--------|--------|\n| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|\n| `Residence Time` | float | Time (in minutes) of the reaction|\n| `Temperature`| float | Temperature (in Celsius) of the reaction|\n| `SM` | float | Quantity of starting material measured (yield %)|\n| `Product 2` | float | Quantity of product 2 measured (yield %)| \n| `Product 3` | float | Quantity of product 3 measured (yield %)| \n| `SOLVENT_NAME` | str | Chemical name of the solvent; used as a key when looking up featurizations| \n| `SOLVENT_RATIO` | list[float] | Ratio of component solvents [1]|\n| `{...} SMILES` | str | SMILES string representation of a molecule|\n\n**Inputs**: `Residence Time`, `Temperature`, `SOLVENT NAME`\n\n**Outputs**: `SM`, `Product 2`, `Product 3` \n\n[1] This is different than the ratios in the solvent ramp experiments. Here, a single solvent has two component molecules, eg. the solvent \"Acetonitrile.Acetic Acid\" has two compounds. The `SOLVENT_RATIO` gives the ratio between these compounds. Most solvents consist of only a single compound, so the ratio will be `[1.0]`."
    },
    {
      "contentUrl": "claisen_data_clean.csv",
      "containedIn": {
        "@id": "archive.zip"
      },
      "encodingFormat": "text/csv",
      "@id": "claisen_data_clean.csv_fileobject",
      "@type": "cr:FileObject",
      "name": "claisen_data_clean.csv",
      "description": "Allyl Phenyl Ether data-set taken from:\n\nLinden Schrecker, Robert McCabe, Jose Pablo Folch, Klaus Hellgardt, King Kuok Hii, Joachim Dickhaut, Christian Holtze, and Andy Wieja. Solvent screening method. https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2025073762, April 2025. WO Patent WO/2025/073762, PCT/EP2024/077742.\n\n| Name | Type | Description |\n|--------|--------|--------|\n| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|\n| `Residence Time` | float | Time (in minutes) of the reaction|\n| `Temperature`| float | Temperature (in Celsius) of the reaction|\n| `SM` | float | Quantity of starting material measured (yield %)|\n| `Product` | float | Quantity of product measured (yield %)| \n| `SM SMILES` | str | SMILES string representation of starting material|\n| `SolventB%` | float | Percent concentration of solvent B; the rest of the solvent is made up of solvent A|\n| `SOLVENT {A/B} NAME` | str | Chemical name of the solvents; used as a key when looking up featurizations|"
    },
    {
      "contentUrl": "smiles_lookup.csv",
      "containedIn": {
        "@id": "archive.zip"
      },
      "encodingFormat": "text/csv",
      "@id": "smiles_lookup.csv_fileobject",
      "@type": "cr:FileObject",
      "name": "smiles_lookup.csv",
      "description": "Translation of solvent names to SMILES representations."
    },
    {
      "contentUrl": "spange_descriptors_lookup.csv",
      "containedIn": {
        "@id": "archive.zip"
      },
      "encodingFormat": "text/csv",
      "@id": "spange_descriptors_lookup.csv_fileobject",
      "@type": "cr:FileObject",
      "name": "spange_descriptors_lookup.csv",
      "description": "Translation of solvent names to the ML-readable representation based on measurable properties of solvents, taken from:\n\nStefan Spange, Nadine Wei\u00df, Caroline H Schmidt, and Katja Schreiter. Reappraisal of empirical\nsolvent polarity scales for organic solvents. Chemistry-Methods, 1(1):42\u201360, 2021."
    }
  ],
  "recordSet": [
    {
      "field": [
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "acs_pca_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT NAME"
            }
          },
          "@id": "acs_pca_descriptors_lookup.csv/SOLVENT+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT NAME",
          "description": "Solvent name"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "acs_pca_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "PC1"
            }
          },
          "@id": "acs_pca_descriptors_lookup.csv/PC1",
          "@type": "cr:Field",
          "name": "PC1",
          "description": "Principle Component 1"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "acs_pca_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "PC2"
            }
          },
          "@id": "acs_pca_descriptors_lookup.csv/PC2",
          "@type": "cr:Field",
          "name": "PC2",
          "description": "Principle Component 2"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "acs_pca_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "PC3"
            }
          },
          "@id": "acs_pca_descriptors_lookup.csv/PC3",
          "@type": "cr:Field",
          "name": "PC3",
          "description": "Principle Component 3"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "acs_pca_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "PC4"
            }
          },
          "@id": "acs_pca_descriptors_lookup.csv/PC4",
          "@type": "cr:Field",
          "name": "PC4",
          "description": "Principle Component 4"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "acs_pca_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "PC5"
            }
          },
          "@id": "acs_pca_descriptors_lookup.csv/PC5",
          "@type": "cr:Field",
          "name": "PC5",
          "description": "Principle Component 5"
        }
      ],
      "@id": "acs_pca_descriptors_lookup.csv",
      "@type": "cr:RecordSet",
      "name": "acs_pca_descriptors_lookup.csv",
      "description": "Translation of solvent names to the ML-readable representation: ACS Solvent Selection Guide's principle component analysis representation (ACS PCA)"
    },
    {
      "field": [
        {
          "dataType": [
            "sc:Integer"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "EXP NUM"
            }
          },
          "@id": "catechol_full_data_yields.csv/EXP+NUM",
          "@type": "cr:Field",
          "name": "EXP NUM",
          "description": "Experiment index; all rows with the same `EXP NUM` will use the same solvent"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT A NAME"
            }
          },
          "@id": "catechol_full_data_yields.csv/SOLVENT+A+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT A NAME",
          "description": "Chemical name of solvent A"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT B NAME"
            }
          },
          "@id": "catechol_full_data_yields.csv/SOLVENT+B+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT B NAME",
          "description": "Chemical name of solvent B"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SolventB%"
            }
          },
          "@id": "catechol_full_data_yields.csv/SolventB%25",
          "@type": "cr:Field",
          "name": "SolventB%",
          "description": "Percent concentration of solvent B; the rest of the solvent is made up of solvent A"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Residence Time"
            }
          },
          "@id": "catechol_full_data_yields.csv/Residence+Time",
          "@type": "cr:Field",
          "name": "Residence Time",
          "description": "Time (in minutes) of the reaction"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Temperature"
            }
          },
          "@id": "catechol_full_data_yields.csv/Temperature",
          "@type": "cr:Field",
          "name": "Temperature",
          "description": "Temperature (in Celsius) of the reaction"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SM"
            }
          },
          "@id": "catechol_full_data_yields.csv/SM",
          "@type": "cr:Field",
          "name": "SM",
          "description": "Quantity of starting material measured (yield %)"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 2"
            }
          },
          "@id": "catechol_full_data_yields.csv/Product+2",
          "@type": "cr:Field",
          "name": "Product 2",
          "description": "Quantity of product 2 measured (yield %)"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 3"
            }
          },
          "@id": "catechol_full_data_yields.csv/Product+3",
          "@type": "cr:Field",
          "name": "Product 3",
          "description": "Quantity of product 3 measured (yield %)"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SM SMILES"
            }
          },
          "@id": "catechol_full_data_yields.csv/SM+SMILES",
          "@type": "cr:Field",
          "name": "SM SMILES",
          "description": "SMILES string representation of starting material"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 2 SMILES"
            }
          },
          "@id": "catechol_full_data_yields.csv/Product+2+SMILES",
          "@type": "cr:Field",
          "name": "Product 2 SMILES",
          "description": "SMILES string representation of product 2"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 3 SMILES"
            }
          },
          "@id": "catechol_full_data_yields.csv/Product+3+SMILES",
          "@type": "cr:Field",
          "name": "Product 3 SMILES",
          "description": "SMILES string representation of product 3"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT A SMILES"
            }
          },
          "@id": "catechol_full_data_yields.csv/SOLVENT+A+SMILES",
          "@type": "cr:Field",
          "name": "SOLVENT A SMILES",
          "description": "SMILES string representation of solvent A"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT B SMILES"
            }
          },
          "@id": "catechol_full_data_yields.csv/SOLVENT+B+SMILES",
          "@type": "cr:Field",
          "name": "SOLVENT B SMILES",
          "description": "SMILES string representation of solvent B"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT A Ratio"
            }
          },
          "@id": "catechol_full_data_yields.csv/SOLVENT+A+Ratio",
          "@type": "cr:Field",
          "name": "SOLVENT A Ratio",
          "description": "Ratio of the component solvents in solvent A"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT B Ratio"
            }
          },
          "@id": "catechol_full_data_yields.csv/SOLVENT+B+Ratio",
          "@type": "cr:Field",
          "name": "SOLVENT B Ratio",
          "description": "Ratio of the component solvents in solvent B"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Reaction SMILES A"
            }
          },
          "@id": "catechol_full_data_yields.csv/Reaction+SMILES+A",
          "@type": "cr:Field",
          "name": "Reaction SMILES A",
          "description": "SMILES string representation of reaction between starting material and Solvent A"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "Reaction SMILES B"
            }
          },
          "@id": "catechol_full_data_yields.csv/Reaction+SMILES+B",
          "@type": "cr:Field",
          "name": "Reaction SMILES B",
          "description": "SMILES string representation of reaction between starting material and Solvent B"
        },
        {
          "dataType": [
            "sc:Integer"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_full_data_yields.csv_fileobject"
            },
            "extract": {
              "column": "RAMP NUM"
            }
          },
          "@id": "catechol_full_data_yields.csv/RAMP+NUM",
          "@type": "cr:Field",
          "name": "RAMP NUM",
          "description": "Ramp number"
        }
      ],
      "@id": "catechol_full_data_yields.csv",
      "@type": "cr:RecordSet",
      "name": "catechol_full_data_yields.csv",
      "description": "Full data set with mixture solvents.\n\n| Name | Type | Description |\n|--------|--------|--------|\n| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|\n| `SOLVENT {A/B} NAME` | str | Chemical name of the solvents; used as a key when looking up featurizations|\n| `SolventB%` | float | Percent concentration of solvent B; the rest of the solvent is made up of solvent A|\n| `Residence Time` | float | Time (in minutes) of the reaction|\n| `Temperature`| float | Temperature (in Celsius) of the reaction|\n| `SM` | float | Quantity of starting material measured (yield %)|\n| `Product {2/3}` | float | Quantity of product {2/3} measured (yield %)| \n| `SM SMILES` | str | SMILES string representation of starting material|\n| `Product {2/3} SMILES` | str | SMILES string representation of product 2/3|\n| `SOLVENT {A/B} SMILES` | str | SMILES string representation of solvent {A/B}|\n| `SOLVENT {A/B} Ratio` | str |  Ratio of the component solvents in solvent {A/B} |\n| `Reaction SMILES {A/B}` | str | SMILES string representation of reaction between starting material and Solvent {A/B}  |\n| `RAMP NUM` | str | Ramp number |\n\n\n\n**Inputs:** Residence Time, Temperature, SOLVENT A NAME, SOLVENT B NAME, SolventB%\n**Outputs:** SM, Product 2, Product 3"
    },
    {
      "field": [
        {
          "dataType": [
            "sc:Integer"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "EXP NUM"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/EXP+NUM",
          "@type": "cr:Field",
          "name": "EXP NUM"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "Residence Time"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/Residence+Time",
          "@type": "cr:Field",
          "name": "Residence Time"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "Temperature"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/Temperature",
          "@type": "cr:Field",
          "name": "Temperature"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "SM"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/SM",
          "@type": "cr:Field",
          "name": "SM"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 2"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/Product+2",
          "@type": "cr:Field",
          "name": "Product 2"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 3"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/Product+3",
          "@type": "cr:Field",
          "name": "Product 3"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "SM SMILES"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/SM+SMILES",
          "@type": "cr:Field",
          "name": "SM SMILES"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 2 SMILES"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/Product+2+SMILES",
          "@type": "cr:Field",
          "name": "Product 2 SMILES"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "Product 3 SMILES"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/Product+3+SMILES",
          "@type": "cr:Field",
          "name": "Product 3 SMILES"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT NAME"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/SOLVENT+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT NAME"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT SMILES"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/SOLVENT+SMILES",
          "@type": "cr:Field",
          "name": "SOLVENT SMILES"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT Ratio"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/SOLVENT+Ratio",
          "@type": "cr:Field",
          "name": "SOLVENT Ratio"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "catechol_single_solvent_yields.csv_fileobject"
            },
            "extract": {
              "column": "Reaction SMILES"
            }
          },
          "@id": "catechol_single_solvent_yields.csv/Reaction+SMILES",
          "@type": "cr:Field",
          "name": "Reaction SMILES"
        }
      ],
      "@id": "catechol_single_solvent_yields.csv",
      "@type": "cr:RecordSet",
      "name": "catechol_single_solvent_yields.csv",
      "description": "Single-solvent dataset. \n\n| Name | Type | Description |\n|--------|--------|--------|\n| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|\n| `Residence Time` | float | Time (in minutes) of the reaction|\n| `Temperature`| float | Temperature (in Celsius) of the reaction|\n| `SM` | float | Quantity of starting material measured (yield %)|\n| `Product 2` | float | Quantity of product 2 measured (yield %)| \n| `Product 3` | float | Quantity of product 3 measured (yield %)| \n| `SOLVENT_NAME` | str | Chemical name of the solvent; used as a key when looking up featurizations| \n| `SOLVENT_RATIO` | list[float] | Ratio of component solvents [1]|\n| `{...} SMILES` | str | SMILES string representation of a molecule|\n\n**Inputs**: `Residence Time`, `Temperature`, `SOLVENT NAME`\n\n**Outputs**: `SM`, `Product 2`, `Product 3` \n\n[1] This is different than the ratios in the solvent ramp experiments. Here, a single solvent has two component molecules, eg. the solvent \"Acetonitrile.Acetic Acid\" has two compounds. The `SOLVENT_RATIO` gives the ratio between these compounds. Most solvents consist of only a single compound, so the ratio will be `[1.0]`."
    },
    {
      "field": [
        {
          "dataType": [
            "sc:Integer"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "EXP NUM"
            }
          },
          "@id": "claisen_data_clean.csv/EXP+NUM",
          "@type": "cr:Field",
          "name": "EXP NUM",
          "description": "Experiment index; all rows with the same `EXP NUM` will use the same solvent"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT A NAME"
            }
          },
          "@id": "claisen_data_clean.csv/SOLVENT+A+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT A NAME",
          "description": "Chemical name of solvent A"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT B NAME"
            }
          },
          "@id": "claisen_data_clean.csv/SOLVENT+B+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT B NAME",
          "description": "Chemical name of solvent B"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SolventB%"
            }
          },
          "@id": "claisen_data_clean.csv/SolventB%25",
          "@type": "cr:Field",
          "name": "SolventB%",
          "description": "Percent concentration of solvent B; the rest of the solvent is made up of solvent A"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "Residence Time"
            }
          },
          "@id": "claisen_data_clean.csv/Residence+Time",
          "@type": "cr:Field",
          "name": "Residence Time",
          "description": "Time (in minutes) of the reaction"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "Temperature"
            }
          },
          "@id": "claisen_data_clean.csv/Temperature",
          "@type": "cr:Field",
          "name": "Temperature",
          "description": "Temperature (in Celsius) of the reaction"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SM"
            }
          },
          "@id": "claisen_data_clean.csv/SM",
          "@type": "cr:Field",
          "name": "SM",
          "description": "Quantity of starting material measured (yield %)"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "Product"
            }
          },
          "@id": "claisen_data_clean.csv/Product",
          "@type": "cr:Field",
          "name": "Product",
          "description": "Quantity of product measured (yield %)"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SM SMILES"
            }
          },
          "@id": "claisen_data_clean.csv/SM+SMILES",
          "@type": "cr:Field",
          "name": "SM SMILES",
          "description": "SMILES string representation of starting material"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "Product SMILES"
            }
          },
          "@id": "claisen_data_clean.csv/Product+SMILES",
          "@type": "cr:Field",
          "name": "Product SMILES",
          "description": "SMILES string representation of product"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT A SMILES"
            }
          },
          "@id": "claisen_data_clean.csv/SOLVENT+A+SMILES",
          "@type": "cr:Field",
          "name": "SOLVENT A SMILES"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT B SMILES"
            }
          },
          "@id": "claisen_data_clean.csv/SOLVENT+B+SMILES",
          "@type": "cr:Field",
          "name": "SOLVENT B SMILES"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT A Ratio"
            }
          },
          "@id": "claisen_data_clean.csv/SOLVENT+A+Ratio",
          "@type": "cr:Field",
          "name": "SOLVENT A Ratio"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT B Ratio"
            }
          },
          "@id": "claisen_data_clean.csv/SOLVENT+B+Ratio",
          "@type": "cr:Field",
          "name": "SOLVENT B Ratio"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "Reaction SMILES A"
            }
          },
          "@id": "claisen_data_clean.csv/Reaction+SMILES+A",
          "@type": "cr:Field",
          "name": "Reaction SMILES A"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "claisen_data_clean.csv_fileobject"
            },
            "extract": {
              "column": "Reaction SMILES B"
            }
          },
          "@id": "claisen_data_clean.csv/Reaction+SMILES+B",
          "@type": "cr:Field",
          "name": "Reaction SMILES B"
        }
      ],
      "@id": "claisen_data_clean.csv",
      "@type": "cr:RecordSet",
      "name": "claisen_data_clean.csv",
      "description": "Allyl Phenyl Ether data-set taken from:\n\nLinden Schrecker, Robert McCabe, Jose Pablo Folch, Klaus Hellgardt, King Kuok Hii, Joachim Dickhaut, Christian Holtze, and Andy Wieja. Solvent screening method. https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2025073762, April 2025. WO Patent WO/2025/073762, PCT/EP2024/077742.\n\n| Name | Type | Description |\n|--------|--------|--------|\n| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|\n| `Residence Time` | float | Time (in minutes) of the reaction|\n| `Temperature`| float | Temperature (in Celsius) of the reaction|\n| `SM` | float | Quantity of starting material measured (yield %)|\n| `Product` | float | Quantity of product measured (yield %)| \n| `SM SMILES` | str | SMILES string representation of starting material|\n| `SolventB%` | float | Percent concentration of solvent B; the rest of the solvent is made up of solvent A|\n| `SOLVENT {A/B} NAME` | str | Chemical name of the solvents; used as a key when looking up featurizations|"
    },
    {
      "field": [
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "smiles_lookup.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT NAME"
            }
          },
          "@id": "smiles_lookup.csv/SOLVENT+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT NAME",
          "description": "Solvent name"
        },
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "smiles_lookup.csv_fileobject"
            },
            "extract": {
              "column": "solvent smiles"
            }
          },
          "@id": "smiles_lookup.csv/solvent+smiles",
          "@type": "cr:Field",
          "name": "solvent smiles",
          "description": "SMILES representation"
        }
      ],
      "@id": "smiles_lookup.csv",
      "@type": "cr:RecordSet",
      "name": "smiles_lookup.csv",
      "description": "Translation of solvent names to SMILES representations."
    },
    {
      "field": [
        {
          "dataType": [
            "sc:Text"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "SOLVENT NAME"
            }
          },
          "@id": "spange_descriptors_lookup.csv/SOLVENT+NAME",
          "@type": "cr:Field",
          "name": "SOLVENT NAME",
          "description": "solvent name"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "dielectric constant"
            }
          },
          "@id": "spange_descriptors_lookup.csv/dielectric+constant",
          "@type": "cr:Field",
          "name": "dielectric constant",
          "description": "Dielectric constant of the solvent"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "ET(30)"
            }
          },
          "@id": "spange_descriptors_lookup.csv/ET(30)",
          "@type": "cr:Field",
          "name": "ET(30)",
          "description": "Reichardt's solvent polarity parameter"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "alpha"
            }
          },
          "@id": "spange_descriptors_lookup.csv/alpha",
          "@type": "cr:Field",
          "name": "alpha",
          "description": "hydrogen bond donating ability"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "beta"
            }
          },
          "@id": "spange_descriptors_lookup.csv/beta",
          "@type": "cr:Field",
          "name": "beta",
          "description": "hydrogen bond accepting ability"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "pi*"
            }
          },
          "@id": "spange_descriptors_lookup.csv/pi*",
          "@type": "cr:Field",
          "name": "pi*",
          "description": "Kamelt-Taft dipolarity parameter"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "SA"
            }
          },
          "@id": "spange_descriptors_lookup.csv/SA",
          "@type": "cr:Field",
          "name": "SA",
          "description": "solvent acidity"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "SB"
            }
          },
          "@id": "spange_descriptors_lookup.csv/SB",
          "@type": "cr:Field",
          "name": "SB",
          "description": "solvent basicity"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "SP"
            }
          },
          "@id": "spange_descriptors_lookup.csv/SP",
          "@type": "cr:Field",
          "name": "SP",
          "description": "solvent dipolarity"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "SdP"
            }
          },
          "@id": "spange_descriptors_lookup.csv/SdP",
          "@type": "cr:Field",
          "name": "SdP"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "N"
            }
          },
          "@id": "spange_descriptors_lookup.csv/N",
          "@type": "cr:Field",
          "name": "N",
          "description": "molar concentration"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "n"
            }
          },
          "@id": "spange_descriptors_lookup.csv/n",
          "@type": "cr:Field",
          "name": "n",
          "description": "diffraction index"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "f(n)"
            }
          },
          "@id": "spange_descriptors_lookup.csv/f(n)",
          "@type": "cr:Field",
          "name": "f(n)",
          "description": "onsager function f(n) = (n^2 - 1) / (n^2 + 2), where n is diffraction index"
        },
        {
          "dataType": [
            "sc:Float"
          ],
          "source": {
            "fileObject": {
              "@id": "spange_descriptors_lookup.csv_fileobject"
            },
            "extract": {
              "column": "delta"
            }
          },
          "@id": "spange_descriptors_lookup.csv/delta",
          "@type": "cr:Field",
          "name": "delta",
          "description": "hildebrand solubility parameter"
        }
      ],
      "@id": "spange_descriptors_lookup.csv",
      "@type": "cr:RecordSet",
      "name": "spange_descriptors_lookup.csv",
      "description": "Translation of solvent names to the ML-readable representation based on measurable properties of solvents, taken from:\n\nStefan Spange, Nadine Wei\u00df, Caroline H Schmidt, and Katja Schreiter. Reappraisal of empirical\nsolvent polarity scales for organic solvents. Chemistry-Methods, 1(1):42\u201360, 2021."
    }
  ],
  "version": 4,
  "keywords": [
    "subject > earth and nature > physical science > chemistry",
    "task > regression",
    "technique > transfer learning",
    "subject > earth and nature",
    "subject > people and society > business > finance > investing"
  ],
  "isAccessibleForFree": true,
  "includedInDataCatalog": {
    "@type": "sc:DataCatalog",
    "name": "Kaggle",
    "url": "https://www.kaggle.com"
  },
  "creator": {
    "@type": "sc:Organization",
    "name": "AIchemy",
    "url": "/organizations/aichemy",
    "image": "https://storage.googleapis.com/kaggle-organizations/5139/thumbnail.png?t=2025-05-09-08-19-46"
  },
  "publisher": {
    "@type": "sc:Organization",
    "name": "Kaggle",
    "url": "https://www.kaggle.com/organizations/kaggle",
    "image": "https://storage.googleapis.com/kaggle-organizations/4/thumbnail.png"
  },
  "thumbnailUrl": "https://storage.googleapis.com/kaggle-datasets-images/7372654/11744485/ecbca69f1c493d0563ec63911e4eff06/dataset-card.png?t=2025-05-09-08-43-06",
  "dateModified": "2025-05-15T13:10:10.333",
  "datePublished": "2025-05-15T12:47:06.9090487",
  "@type": "sc:Dataset",
  "name": "Catechol Benchmark",
  "url": "https://www.kaggle.com/datasets/aichemy/catechol-benchmark/versions/4",
  "description": "# Summary \n\nCatechol dataset for solvent selection and machine learning.\n\n_A link to the accompanying publication will be provided here as soon as possible_\n\n# Data files\n\n## Main data files\n\n- **catechol_full_data_yields.csv**: Full data set with mixture solvents\n- **catechol_single_solvent_yields.csv**: Only the single-solvent data\n- **claisen_data_clean.csv**: Allyl Phenyl Ether data-set from an external source\n\n## Lookup tables\n\nTables translating solvent names - as tabulated the main data files - to various pre-computed ML-readable representations:\n\n- **acs_pca_descriptors_lookup.csv**: ACS Solvent Selection Guide's principle component analysis representation.\n- **drfps_lookup.csv**: Fingerprint representation created using the difference in sets containing molecular substructures to the left and right of the reaction arrow in a SMILES string\n- **fragprints_lookup.csv**: fragprints: A combination of molecular fingerprints, which are bit vectors indicating the presence of substructures in the molecule, and molecular fragments, which are count vectors indicating the number of times specific functional groups appear.\n- **spange_descriptors_lookup.csv:** Representation based on measurable properties of solvents\n- **smiles_lookup.csv**: SMILES strings. \n"
}
```