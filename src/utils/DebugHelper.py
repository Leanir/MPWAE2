import numpy as np
from MPWAE2.src.classes.MPWAE2 import MPWAE2
from torch.optim               import Adam
from torch.nn                  import MSELoss, Conv1d, ConvTranspose1d, Module
from torch                     import device, amp, Tensor, isnan, isinf, no_grad
from typing                    import Tuple, Dict, List, Optional
from torch.utils.data          import DataLoader
from pandas                    import DataFrame, Series


# region Debugging class
class DebugHelper:
    def __init__(self,
            model: MPWAE2,
            optimizer: Adam,
            loss_func: MSELoss,
            dev: device):
        self.model     = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device    = dev

        self.sparse_layer_types = ('SparseLinear',)


    def count_active_connections(self) -> Tuple[int, int]:
        """Count active gradient connections in sparse layers."""
        total_active = 0
        total_possible = 0

        for _, module in self.model.named_modules():
            cond = self._is_sparse_layer(module) \
                and hasattr(module, 'weight')    \
                and module.weight.grad is not None
            if cond:
                # Count non-zero gradients where mask is True
                masked_grads = module.weight.grad * module.mask
                active_grads = (masked_grads != 0).sum().item()
                possible_grads = module.mask.sum().item()

                total_active += active_grads
                total_possible += possible_grads

        return total_active, total_possible


    def analyze_gradient_flow(self) -> Dict[str, Dict[str, float]]:
        print("=== Gradient Flow Analysis ===")
        gradient_stats = {}

        # Analyze sparse layers
        for name, module in self.model.named_modules():
            if self._is_sparse_layer(module):
                stats = self._analyze_sparse_layer_gradients(name, module)
            elif isinstance(module, (Conv1d, ConvTranspose1d)):
                stats = self._analyze_conv_layer_gradients(name, module)
            else:
                stats = None

            if stats:
                gradient_stats[name] = stats

        return gradient_stats


    def monitor_sparse_training(self,
            epoch: int,
            batch_idx: int,
            log_frequency: int = 20) -> None:

        if batch_idx % log_frequency == 0:
            print(f"\n--- Epoch {epoch+1}, Batch {batch_idx} ---")

            # Count active connections
            active, possible = self.count_active_connections()
            if possible > 0:
                print(
                    f"Active gradient connections: {active}/{possible}\t",
                    f"({100*active/possible:.2f}%)"
                )
            else:
                print("No sparse connections found   ╭∩╮(-_-)╭∩╮")

            # gradient health check
            problematic_layers = self._check_gradient_health()

            if problematic_layers:
                print("(ᗒᗣᗕ)  Problematic gradients:")
                for layer_info in problematic_layers:
                    print(f"\t{layer_info}")
            else:
                print("Gradients look healthy    \ (•◡•) /")

            print("---")


    def test_gradient_computation(self, data_loader: DataLoader) -> None:
        print("=== Testing Gradient Computation ===")
        self.model.train()

        # grab one batch
        test_batch = next(iter(data_loader)).to(self.device)

        # clear gradients
        self.optimizer.zero_grad()

        # forward pass
        dev_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with amp.autocast(device_type=dev_type):
            output = self.model(test_batch)
            loss = self.loss_func(output, test_batch)

        print(f"Test loss: {loss.item():.6f}")

        # backward pass
        loss.backward()

        # check gradients
        print("Checking gradients after backward():")
        active, possible = self.count_active_connections()
        print(f"Active connections: {active}/{possible}")

        if active == 0:
            print("Still no gradients. Checking parameter details")
            self._check_parameter_gradients()


    def debug_nan_issues(self, data_loader: DataLoader) -> bool:
        print("=== DEBUGGING NaN ISSUES ===")
        self.model.train()

        # single batch
        test_batch = next(iter(data_loader)).to(self.device)

        # Check input
        nan_found = self._check_input_data(test_batch)
        if nan_found:
            return True

        # Clear gradients
        self.optimizer.zero_grad()

        # Forward pass step by step
        print("\n--- Forward Pass Debug ---")
        nan_found = self._debug_forward_pass(test_batch)

        if nan_found:
            return True

        # Check model parameters for NaN
        print("\n--- Parameter Check ---")
        nan_found = self._check_model_parameters()

        return nan_found


    def debug_data_pipeline(self,
            tumor_df: DataFrame,
            train_sample_ids: List[str]) -> Tuple[Series, Series, Series]:
        print("=== DATA PIPELINE DEBUG ===")

        # Check original tumor_df
        print("\n1. Original tumor_df:")
        self._analyze_dataframe(tumor_df, "Original")

        # Check train_df (subset)
        print("\n2. Training subset (train_df):")
        train_df = tumor_df[train_sample_ids]
        self._analyze_dataframe(train_df, "Training subset")

        # Check min/max calculation
        print("\n3. Min/Max calculation:")
        min_vals, max_vals = self._analyze_minmax_calculation(train_df)

        # Check if min == max (causes division by zero)
        print("\n4. Division by zero check:")
        diff = self._check_division_by_zero(min_vals, max_vals)

        return min_vals, max_vals, diff


    def debug_model_architecture(self) -> None:
        print("=== MODEL ARCHITECTURE DEBUG ===")

        # Check model structure
        print("Model structure:")
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                print(
                    f"\t{name}: {type(module).__name__} - {module.weight.shape}")

        # Check for dimension mismatches in convolutional layers
        print("\nConvolutional layer analysis:")
        for name, module in self.model.named_modules():
            if isinstance(module, (Conv1d, ConvTranspose1d)):
                print(f"\t{name}: {type(module).__name__}")
                print(f"\t\tInput channels:  {module.in_channels}")
                print(f"\t\tOutput channels: {module.out_channels}")
                print(f"\t\tKernel size:     {module.kernel_size}")
                print(f"\t\tStride:          {module.stride}")
                print(f"\t\tPadding:         {module.padding}")

    # Private helper methods

    def _is_sparse_layer(self, module: Module) -> bool:
        return any(
            sparse_type in str(type(module)) \
            for sparse_type in self.sparse_layer_types
        )


    def _analyze_sparse_layer_gradients(self,
            name: str,
            module: Module) -> Optional[Dict[str, float]]:
        if hasattr(module, 'weight') and module.weight.grad is not None:
            # Gradient statistics
            grad_norm = module.weight.grad.norm().item()
            masked_grad = module.weight.grad * module.mask
            masked_grad_norm = masked_grad.norm().item()

            # Count active gradients
            active_grads = (masked_grad != 0).sum().item()
            total_connections = module.mask.sum().item()

            stats = {
                'full_grad_norm': grad_norm,
                'masked_grad_norm': masked_grad_norm,
                'active_grads': active_grads,
                'total_connections': total_connections,
                'grad_min': masked_grad.min().item(),
                'grad_max': masked_grad.max().item()
            }

            print(f"{name}:")
            print(f"\tFull gradient norm: {grad_norm:.6f}")
            print(f"\tMasked gradient norm: {masked_grad_norm:.6f}")
            print(f"\tActive gradients: {active_grads}/{total_connections} ",
                  f"({100*active_grads/total_connections:.1f}%)")
            print(
                f"\tGradient range: [{stats['grad_min']:.6f}, {stats['grad_max']:.6f}]\n")

            return stats
        else:
            print(f"{name}: No gradients computed")
            return None


    def _analyze_conv_layer_gradients(self,
            name: str,
            module: Module) -> Optional[Dict[str, float]]:
        if hasattr(module, 'weight') and module.weight.grad is not None:
            grad_norm = module.weight.grad.norm().item()
            print(f"{name} (Conv): gradient norm = {grad_norm:.6f}")

            return {'grad_norm': grad_norm}

        return None


    def _check_gradient_health(self) -> List[str]:
        problematic_layers = []

        for name, module in self.model.named_modules():
            cond = self._is_sparse_layer(module) \
                and hasattr(module, 'weight')    \
                and module.weight.grad is not None

            if cond:
                masked_grad = module.weight.grad * module.mask
                grad_norm = masked_grad.norm().item()

                if grad_norm > 10.0:
                    problematic_layers.append(
                        f"{name}: exploding ({grad_norm:.2f})")
                elif grad_norm < 1e-6:
                    problematic_layers.append(
                        f"{name}: vanishing ({grad_norm:.2e})")

        return problematic_layers


    def _check_parameter_gradients(self) -> None:
        for name, param in self.model.named_parameters():
            print(
                f"{name}: requires_grad={param.requires_grad}, ",
                f"grad={'exists' if param.grad is not None else 'None'}"
            )


    def _check_input_data(self, test_batch: Tensor) -> bool:
        print(f"Input batch shape: {test_batch.shape}")
        print(f"""Input batch stats:
            min  = {test_batch.min().item():.6f}
            max  = {test_batch.max().item():.6f}
            mean = {test_batch.mean().item():.6f}
        """)

        has_nan = isnan(test_batch).any().item()
        has_inf = isinf(test_batch).any().item()

        print(f"Input has NaN: {has_nan}")
        print(f"Input has Inf: {has_inf}")

        return has_nan or has_inf


    def _debug_forward_pass(self, test_batch: Tensor) -> bool:
        try:
            with no_grad():
                x = test_batch
                print(f"""
                    Input shape = {x.shape},
                    Range       = [{x.min().item():.6f}, {x.max().item():.6f}]
                """)

                # Debug encoder
                if hasattr(self.model, 'encoder'):
                    print("Debugging encoder...")
                    x = self._debug_encoder_forward(x)
                    if isnan(x).any():
                        print("!!! NaN detected in encoder - stopping debug !!!")
                        return True

                # Debug decoder
                if hasattr(self.model, 'decoder'):
                    print("Debugging decoder...")
                    x = self._debug_decoder_forward(x)
                    if isnan(x).any():
                        print("!!! NaN detected in decoder - stopping debug !!!")
                        return True

                print(f"""
                    Output shape = {x.shape},
                    Range        = [{x.min().item():.6f}, {x.max().item():.6f}]
                """)

        except Exception as e:
            print(f"[ERROR] during forward pass: {e}")
            return True

        return False


    def _debug_encoder_forward(self, x: Tensor) -> Tensor:
        encoder = self.model.encoder

        # First sparse layer
        if hasattr(encoder, 'in_hid'):
            x = encoder.in_hid(x)
            self._log_tensor_stats(x, "After in_hid")
            if isnan(x).any():
                return x

        # Activation
        if hasattr(encoder, 'act'):
            x = encoder.act(x)
            self._log_tensor_stats(x, "After first activation")
            if isnan(x).any():
                return x

        # Second sparse layer
        if hasattr(encoder, 'hid_cnv1'):
            x = encoder.hid_cnv1(x)
            self._log_tensor_stats(x, "After hid_cnv1")
            if isnan(x).any():
                return x

        # Another activation
        if hasattr(encoder, 'act'):
            x = encoder.act(x)
            self._log_tensor_stats(x, "After second activation")
            if isnan(x).any():
                return x

        # Unsqueeze for conv layers
        x = x.unsqueeze(1)
        self._log_tensor_stats(x, "After unsqueeze")

        # First conv layer
        if hasattr(encoder, 'cnv1_cnv2'):
            x = encoder.act(encoder.cnv1_cnv2(x))
            self._log_tensor_stats(x, "After cnv1_cnv2")
            if isnan(x).any():
                return x

        # Second conv layer
        if hasattr(encoder, 'cnv2_out'):
            x = encoder.act(encoder.cnv2_out(x))
            self._log_tensor_stats(x, "After cnv2_out")
            if isnan(x).any():
                return x

        # Squeeze back
        x = x.squeeze(1)
        self._log_tensor_stats(x, "After squeeze")

        return x


    def _debug_decoder_forward(self, x: Tensor) -> Tensor:
        """Debug decoder forward pass."""
        decoder = self.model.decoder

        # Unsqueeze for transpose conv layers
        x = x.unsqueeze(1)
        self._log_tensor_stats(x, "After unsqueeze")

        # First transpose conv layer
        if hasattr(decoder, 'emb_dcnv1'):
            x = decoder.act(decoder.emb_dcnv1(x))
            self._log_tensor_stats(x, "After emb_dcnv1")
            if isnan(x).any():
                return x

        # Second transpose conv layer
        if hasattr(decoder, 'dcnv1_dcnv2'):
            x = decoder.act(decoder.dcnv1_dcnv2(x))
            self._log_tensor_stats(x, "After dcnv1_dcnv2")
            if isnan(x).any():
                return x

        # Squeeze back
        x = x.squeeze(1)
        self._log_tensor_stats(x, "After squeeze")

        # First sparse layer
        if hasattr(decoder, 'dcnv2_hid'):
            x = decoder.dcnv2_hid(x)
            self._log_tensor_stats(x, "After dcnv2_hid")
            if isnan(x).any():
                return x

        # Activation
        if hasattr(decoder, 'act'):
            x = decoder.act(x)
            self._log_tensor_stats(x, "After first activation")
            if isnan(x).any():
                return x

        # Second sparse layer
        if hasattr(decoder, 'hid_out'):
            x = decoder.hid_out(x)
            self._log_tensor_stats(x, "After hid_out")
            if isnan(x).any():
                return x

        # Final activation
        if hasattr(decoder, 'act'):
            x = decoder.act(x)
            self._log_tensor_stats(x, "After final activation")

        return x


    def _log_tensor_stats(self, tensor: Tensor, label: str) -> None:
        print(f"""
            {label} shape: {tensor.shape}
            range   = [{tensor.min().item():.6f}, {tensor.max().item():.6f}]
            has_nan = {isnan(tensor).any().item()}
        """)

    def _check_model_parameters(self) -> bool:
        nan_found = False

        for name, param in self.model.named_parameters():
            has_nan = isnan(param).any().item()
            has_inf = isinf(param).any().item()

            if has_nan or has_inf:
                print(f"!!! {name}: NaN={has_nan}, Inf={has_inf}")
                nan_found = True
            else:
                print(f"""
                    {name}: OK
                    range=[{param.min().item():.6f}, {param.max().item():.6f}])
                """)

        return nan_found


    def _analyze_dataframe(self, df: DataFrame, label: str) -> None:
        print(f"\t{label} shape: {df.shape}")
        print(f"\tHas NaN: {df.isna().any().any()}")
        print(f"\tHas Inf: {np.isinf(df.values).any()}")

        if df.isna().any().any():
            nan_count = df.isna().sum().sum()
            print(f"\tTotal NaN values: {nan_count}")
            print(f"\tPercentage: {100*nan_count/df.size:.2f}%")


    def _analyze_minmax_calculation(self,
                                    train_df: DataFrame) -> Tuple[Series, Series]:
        min_vals = train_df.min(axis=1)
        max_vals = train_df.max(axis=1)

        print(f"\tMin values shape: {min_vals.shape}")
        print(f"\tMax values shape: {max_vals.shape}")
        print(f"\tMin has NaN: {min_vals.isna().any()}")
        print(f"\tMax has NaN: {max_vals.isna().any()}")
        print(f"\tMin has Inf: {np.isinf(min_vals.values).any()}")
        print(f"\tMax has Inf: {np.isinf(max_vals.values).any()}")

        return min_vals, max_vals


    def _check_division_by_zero(self,
                                min_vals: Series,
                                max_vals: Series) -> Series:
        diff = max_vals - min_vals
        zero_diff = (diff == 0).sum()

        print(f"\tRows where max == min: {zero_diff}")
        print(f"\tThis causes division by zero in normalization!")

        if zero_diff > 0:
            print("\tFirst few problematic rows:")
            problematic_rows = diff[diff == 0].head()
            for idx, val in problematic_rows.items():
                print(
                    f"\t\tRow {idx}: min={min_vals[idx]}, max={max_vals[idx]}, diff={val}")

        return diff
# endregion