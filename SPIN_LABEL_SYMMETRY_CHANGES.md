# Spin label symmetry: line-by-line change log

Branch: `spin-label-symmetry` (local only, created from `tie-spin-field-weights` @ 241883f).
Date: 2026-11-11. Files changed: 2. Lines removed: 4. Lines added: 104.

Each changed line is shown with its line number: `[old N]` = line number in the file
before the change (removed lines), `[new N]` = line number after the change (added lines).
Unchanged lines are not listed.

Purpose: make PolarMACE exactly symmetric under swapping alpha and beta electrons, while
keeping every weight shape so `polar-1-m` can still be loaded for fine-tuning. The
pretrained maps always see the majority spin as alpha; when N_alpha == N_beta both
labelings are averaged.

### `mace/modules/extensions.py`


**Inserted after old line 68 → new lines 69–69**
```diff
+ [new   69]     mix_labelings,
```

**Inserted after old line 1096 → new lines 1098–1106**
```diff
+ [new 1098]         # Which labeling the pretrained maps should see, per atom:
+ [new 1099]         # 1.0 if alpha is the majority spin (the convention used in training),
+ [new 1100]         # 0.0 if beta is (the maps then see the labels swapped),
+ [new 1101]         # 0.5 if N_alpha == N_beta (average both labelings).
+ [new 1102]         spin_excess = data["total_spin"] - 1  # N_alpha - N_beta per graph
+ [new 1103]         label_weight = (0.5 + 0.5 * torch.sign(spin_excess))[data["batch"]].to(
+ [new 1104]             vectors.dtype
+ [new 1105]         )
+ [new 1106] 
```

**Old lines 1135–1135 → new lines 1145–1145**
```diff
- [old 1135]             spin_charge_sources = lr_src(node_feats).squeeze(-2)
+ [new 1145]             spin_charge_sources = lr_src(node_feats, label_weight).squeeze(-2)
```

**Old lines 1183–1183 → new lines 1193–1197**
```diff
- [old 1183]         fukui_sources = self.fukui_source_map(fukui_input)
+ [new 1193]         fukui_sources = self.fukui_source_map(fukui_input)  # [n_nodes, 2]
+ [new 1194]         # Inputs are spin-blind, so swapping labels just swaps the two columns.
+ [new 1195]         fukui_sources = mix_labelings(
+ [new 1196]             fukui_sources, fukui_sources.flip(-1), label_weight
+ [new 1197]         )
```

**Inserted after old line 1193 → new lines 1208–1209**
```diff
+ [new 1208]         # total_spin - 1 = N_alpha - N_beta (may be negative). Flipping its
+ [new 1209]         # sign only swaps the alpha and beta outputs.
```

**Inserted after old line 1270 → new lines 1287–1287**
```diff
+ [new 1287]                 label_weight=label_weight,
```

**Inserted after old line 1318 → new lines 1336–1336**
```diff
+ [new 1336]             label_weight=label_weight,
```

### `mace/modules/field_blocks.py`


**Inserted after old line 28 → new lines 29–56**
```diff
+ [new   29] def swap_spin_channels(x: torch.Tensor) -> torch.Tensor:
+ [new   30]     """Swap the two spin channels of a per-atom tensor.
+ [new   31] 
+ [new   32]     x has shape [N, 2*k], laid out as [alpha (k numbers) | beta (k numbers)].
+ [new   33]     Returns [beta | alpha] with the same shape.
+ [new   34]     """
+ [new   35]     n = x.shape[0]
+ [new   36]     return x.reshape(n, 2, -1).flip(1).reshape(n, -1)
+ [new   37] 
+ [new   38] 
+ [new   39] def mix_labelings(
+ [new   40]     as_given: torch.Tensor,
+ [new   41]     swapped_back: torch.Tensor,
+ [new   42]     label_weight: Optional[torch.Tensor],
+ [new   43] ) -> torch.Tensor:
+ [new   44]     """Per-atom weighted mix of two results: w * as_given + (1 - w) * swapped_back.
+ [new   45] 
+ [new   46]     label_weight has shape [N]: 1 if alpha is the majority spin, 0 if beta is,
+ [new   47]     0.5 if they are equal. None means 0.5 for every atom (plain average).
+ [new   48]     """
+ [new   49]     if label_weight is None:
+ [new   50]         return 0.5 * (as_given + swapped_back)
+ [new   51]     w = label_weight
+ [new   52]     while w.dim() < as_given.dim():
+ [new   53]         w = w.unsqueeze(-1)
+ [new   54]     return w * as_given + (1.0 - w) * swapped_back
+ [new   55] 
+ [new   56] 
```

**Old lines 67–67 → new lines 95–97**
```diff
- [old   67]     def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
+ [new   95]     def forward(
+ [new   96]         self, node_feats: torch.Tensor, label_weight: Optional[torch.Tensor] = None
+ [new   97]     ) -> torch.Tensor:
```

**Inserted after old line 72 → new lines 103–104**
```diff
+ [new  103]         # node_feats is spin-blind, so swapping labels just swaps the halves.
+ [new  104]         mp = mix_labelings(mp, swap_spin_channels(mp), label_weight)
```

**Inserted after old line 613 → new lines 646–646**
```diff
+ [new  646]         label_weight: Optional[torch.Tensor] = None,
```

**Old lines 615–615 → new lines 648–675**
```diff
- [old  615]         # create pot feats
+ [new  648]         # Run the learned update with the given labels and with alpha/beta
+ [new  649]         # swapped, then mix them so the result cannot depend on the labels.
+ [new  650]         out = self._update_unsymmetrized(
+ [new  651]             node_attrs, node_feats, potential_features, local_charges
+ [new  652]         )
+ [new  653]         out_swapped = self._update_unsymmetrized(
+ [new  654]             node_attrs,
+ [new  655]             node_feats,
+ [new  656]             swap_spin_channels(potential_features),
+ [new  657]             swap_spin_channels(local_charges),
+ [new  658]         )
+ [new  659]         # Layout of both: [multipoles_alpha | multipoles_beta | fukui_alpha | fukui_beta]
+ [new  660]         # Put each part of out_swapped back into the normal channel order.
+ [new  661]         n_fukui = 2
+ [new  662]         mp_back = swap_spin_channels(out_swapped[:, :-n_fukui])
+ [new  663]         fk_back = swap_spin_channels(out_swapped[:, -n_fukui:])
+ [new  664]         mp = mix_labelings(out[:, :-n_fukui], mp_back, label_weight)
+ [new  665]         fk = mix_labelings(out[:, -n_fukui:], fk_back, label_weight)
+ [new  666]         return torch.cat([mp, fk], dim=-1)
+ [new  667] 
+ [new  668]     def _update_unsymmetrized(
+ [new  669]         self,
+ [new  670]         node_attrs: torch.Tensor,
+ [new  671]         node_feats: torch.Tensor,
+ [new  672]         potential_features: torch.Tensor,
+ [new  673]         local_charges: torch.Tensor,
+ [new  674]     ) -> torch.Tensor:
+ [new  675]         # Original update body (unchanged).
```

**Inserted after old line 769 → new lines 830–830**
```diff
+ [new  830]         label_weight: Optional[torch.Tensor] = None,
```

**Inserted after old line 770 → new lines 832–852**
```diff
+ [new  832]         # Energy with the given labels and with alpha/beta swapped, mixed so the
+ [new  833]         # total cannot depend on which channel is called alpha.
+ [new  834]         e = self._energy_unsymmetrized(
+ [new  835]             node_feats, field_feats, charges_0, charges_induced
+ [new  836]         )
+ [new  837]         e_swapped = self._energy_unsymmetrized(
+ [new  838]             node_feats,
+ [new  839]             swap_spin_channels(field_feats),
+ [new  840]             swap_spin_channels(charges_0),
+ [new  841]             swap_spin_channels(charges_induced),
+ [new  842]         )
+ [new  843]         return mix_labelings(e, e_swapped, label_weight)
+ [new  844] 
+ [new  845]     def _energy_unsymmetrized(
+ [new  846]         self,
+ [new  847]         node_feats: torch.Tensor,
+ [new  848]         field_feats: torch.Tensor,
+ [new  849]         charges_0: torch.Tensor,
+ [new  850]         charges_induced: torch.Tensor,
+ [new  851]     ) -> torch.Tensor:
+ [new  852]         # Original energy body (unchanged).
```

## Full unified diff

```diff
diff --git a/mace/modules/extensions.py b/mace/modules/extensions.py
index 71ef38a..9f6d242 100644
--- a/mace/modules/extensions.py
+++ b/mace/modules/extensions.py
@@ -66,6 +66,7 @@ from .field_blocks import (
     MultiLayerFeatureMixer,
     field_readout_blocks,
     field_update_blocks,
+    mix_labelings,
 )
 from .radial import ZBLBasis
 from .utils import get_edge_vectors_and_lengths, get_symmetric_displacement
@@ -1094,6 +1095,15 @@ class PolarMACE(ScaleShiftMACE):
         else:
             pair_node_energy = torch.zeros_like(node_e0)
 
+        # Which labeling the pretrained maps should see, per atom:
+        # 1.0 if alpha is the majority spin (the convention used in training),
+        # 0.0 if beta is (the maps then see the labels swapped),
+        # 0.5 if N_alpha == N_beta (average both labelings).
+        spin_excess = data["total_spin"] - 1  # N_alpha - N_beta per graph
+        label_weight = (0.5 + 0.5 * torch.sign(spin_excess))[data["batch"]].to(
+            vectors.dtype
+        )
+
         node_es_list: List[torch.Tensor] = []
         node_feats_list: List[torch.Tensor] = []
         spin_charge_density = torch.zeros(
@@ -1132,7 +1142,7 @@ class PolarMACE(ScaleShiftMACE):
             ]
             node_es_list.append(node_es)
 
-            spin_charge_sources = lr_src(node_feats).squeeze(-2)
+            spin_charge_sources = lr_src(node_feats, label_weight).squeeze(-2)
             spin_charge_density = spin_charge_density + spin_charge_sources
 
         node_feats_out = torch.cat(node_feats_list, dim=-1)
@@ -1180,7 +1190,11 @@ class PolarMACE(ScaleShiftMACE):
         fukui_to_mul_ir = getattr(self, "_fukui_to_mul_ir", None)
         if fukui_to_mul_ir is not None:
             fukui_input = fukui_to_mul_ir(fukui_input)
-        fukui_sources = self.fukui_source_map(fukui_input)
+        fukui_sources = self.fukui_source_map(fukui_input)  # [n_nodes, 2]
+        # Inputs are spin-blind, so swapping labels just swaps the two columns.
+        fukui_sources = mix_labelings(
+            fukui_sources, fukui_sources.flip(-1), label_weight
+        )
         fukui_norm = scatter_sum(
             src=safe_double(fukui_sources),
             index=data["batch"],
@@ -1191,6 +1205,8 @@ class PolarMACE(ScaleShiftMACE):
             fukui_norm == 0, torch.ones_like(fukui_norm), fukui_norm
         )
         fukui_sources = fukui_sources / fukui_norm
+        # total_spin - 1 = N_alpha - N_beta (may be negative). Flipping its
+        # sign only swaps the alpha and beta outputs.
         Q_p_S = (data["total_charge"] + (data["total_spin"] - 1))[data["batch"]]
         Q_m_S = (data["total_charge"] - (data["total_spin"] - 1))[data["batch"]]
         pred_total_charges_0 = scatter_sum(
@@ -1268,6 +1284,7 @@ class PolarMACE(ScaleShiftMACE):
                 local_charges=spin_charge_density.view(
                     spin_charge_density.shape[0], -1
                 ),
+                label_weight=label_weight,
             )
 
             current_fukui_sources = charge_sources_out[:, -2:]
@@ -1316,6 +1333,7 @@ class PolarMACE(ScaleShiftMACE):
                 field_independent_spin_charge_density.shape[0], -1
             ),
             charges_induced=spin_charge_density.view(spin_charge_density.shape[0], -1),
+            label_weight=label_weight,
         )
         le_total = scatter_sum(
             src=local_q_e, index=data["batch"], dim=-1, dim_size=num_graphs
diff --git a/mace/modules/field_blocks.py b/mace/modules/field_blocks.py
index 57dd290..45ccf4b 100644
--- a/mace/modules/field_blocks.py
+++ b/mace/modules/field_blocks.py
@@ -26,6 +26,34 @@ from .wrapper_ops import (
 )
 
 
+def swap_spin_channels(x: torch.Tensor) -> torch.Tensor:
+    """Swap the two spin channels of a per-atom tensor.
+
+    x has shape [N, 2*k], laid out as [alpha (k numbers) | beta (k numbers)].
+    Returns [beta | alpha] with the same shape.
+    """
+    n = x.shape[0]
+    return x.reshape(n, 2, -1).flip(1).reshape(n, -1)
+
+
+def mix_labelings(
+    as_given: torch.Tensor,
+    swapped_back: torch.Tensor,
+    label_weight: Optional[torch.Tensor],
+) -> torch.Tensor:
+    """Per-atom weighted mix of two results: w * as_given + (1 - w) * swapped_back.
+
+    label_weight has shape [N]: 1 if alpha is the majority spin, 0 if beta is,
+    0.5 if they are equal. None means 0.5 for every atom (plain average).
+    """
+    if label_weight is None:
+        return 0.5 * (as_given + swapped_back)
+    w = label_weight
+    while w.dim() < as_given.dim():
+        w = w.unsqueeze(-1)
+    return w * as_given + (1.0 - w) * swapped_back
+
+
 @compile_mode("script")
 class MultiLayerFeatureMixer(torch.nn.Module):
     def __init__(
@@ -64,12 +92,16 @@ class EnvironmentDependentSpinSourceBlock(torch.nn.Module):
         self.irreps_out = 2 * o3.Irreps.spherical_harmonics(max_l)
         self.linear = Linear(irreps_in, self.irreps_out, cueq_config=cueq_config)
 
-    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
+    def forward(
+        self, node_feats: torch.Tensor, label_weight: Optional[torch.Tensor] = None
+    ) -> torch.Tensor:
         mp = self.linear(node_feats)
         if self.zero_charges:
             mp_z = torch.zeros_like(mp)
             mp_z[:, 1:] = mp[:, 1:]
             mp = mp_z
+        # node_feats is spin-blind, so swapping labels just swaps the halves.
+        mp = mix_labelings(mp, swap_spin_channels(mp), label_weight)
         return mp.unsqueeze(-2)  # [n_nodes, 1, (max_l+1)^2 * 2]
 
 
@@ -611,8 +643,36 @@ class AgnosticEmbeddedOneBodyVariableUpdate(FieldUpdateBlock):
         edge_index: torch.Tensor,
         potential_features: torch.Tensor,
         local_charges: torch.Tensor,
+        label_weight: Optional[torch.Tensor] = None,
     ) -> torch.Tensor:
-        # create pot feats
+        # Run the learned update with the given labels and with alpha/beta
+        # swapped, then mix them so the result cannot depend on the labels.
+        out = self._update_unsymmetrized(
+            node_attrs, node_feats, potential_features, local_charges
+        )
+        out_swapped = self._update_unsymmetrized(
+            node_attrs,
+            node_feats,
+            swap_spin_channels(potential_features),
+            swap_spin_channels(local_charges),
+        )
+        # Layout of both: [multipoles_alpha | multipoles_beta | fukui_alpha | fukui_beta]
+        # Put each part of out_swapped back into the normal channel order.
+        n_fukui = 2
+        mp_back = swap_spin_channels(out_swapped[:, :-n_fukui])
+        fk_back = swap_spin_channels(out_swapped[:, -n_fukui:])
+        mp = mix_labelings(out[:, :-n_fukui], mp_back, label_weight)
+        fk = mix_labelings(out[:, -n_fukui:], fk_back, label_weight)
+        return torch.cat([mp, fk], dim=-1)
+
+    def _update_unsymmetrized(
+        self,
+        node_attrs: torch.Tensor,
+        node_feats: torch.Tensor,
+        potential_features: torch.Tensor,
+        local_charges: torch.Tensor,
+    ) -> torch.Tensor:
+        # Original update body (unchanged).
         mixed_feats = self.potential_embedding(
             potential_features,
             node_feats,
@@ -767,7 +827,29 @@ class OneBodyMLPFieldReadout(PostScfReadout):
         field_feats: torch.Tensor,
         charges_0: torch.Tensor,
         charges_induced: torch.Tensor,
+        label_weight: Optional[torch.Tensor] = None,
     ):
+        # Energy with the given labels and with alpha/beta swapped, mixed so the
+        # total cannot depend on which channel is called alpha.
+        e = self._energy_unsymmetrized(
+            node_feats, field_feats, charges_0, charges_induced
+        )
+        e_swapped = self._energy_unsymmetrized(
+            node_feats,
+            swap_spin_channels(field_feats),
+            swap_spin_channels(charges_0),
+            swap_spin_channels(charges_induced),
+        )
+        return mix_labelings(e, e_swapped, label_weight)
+
+    def _energy_unsymmetrized(
+        self,
+        node_feats: torch.Tensor,
+        field_feats: torch.Tensor,
+        charges_0: torch.Tensor,
+        charges_induced: torch.Tensor,
+    ) -> torch.Tensor:
+        # Original energy body (unchanged).
         q_in = charges_induced + charges_0
         q_to_mul_ir = getattr(self, "_q_to_mul_ir", None)
         v_to_mul_ir = getattr(self, "_v_to_mul_ir", None)
```
