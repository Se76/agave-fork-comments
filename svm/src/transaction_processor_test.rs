// #[cfg(feature = "dev-context-only-utils")]
// use qualifier_attr::{field_qualifiers, qualifiers};
// use std::thread::sleep;
use {
    crate::transaction_processor::TransactionBatchProcessor,
    solana_clock::{Epoch, Slot}, solana_program_runtime::{loaded_programs::{ForkGraph, ProgramCache}, sysvar_cache::SysvarCache}, solana_pubkey::Pubkey, std::{collections::HashSet, sync::{Arc, RwLock}}
};


pub fn new_uninitialized<FG: ForkGraph>(slot: Slot, epoch: Epoch) -> TransactionBatchProcessor<FG> {
    TransactionBatchProcessor::new_uninitialized(slot, epoch)
}




// #[cfg_attr(feature = "frozen-abi", derive(AbiExample))]
// #[cfg_attr(
//     feature = "dev-context-only-utils",
//     field_qualifiers(slot(pub), epoch(pub))
// )]
// pub struct TransactionBatchProcessor<FG: ForkGraph> {
//     /// Bank slot (i.e. block)
//     slot: Slot,

//     /// Bank epoch
//     epoch: Epoch,

//     /// SysvarCache is a collection of system variables that are
//     /// accessible from on chain programs. It is passed to SVM from
//     /// client code (e.g. Bank) and forwarded to the MessageProcessor.
//     sysvar_cache: RwLock<SysvarCache>,

//     /// Programs required for transaction batch processing
//     pub program_cache: Arc<RwLock<ProgramCache<FG>>>,

//     /// Builtin program ids
//     pub builtin_program_ids: RwLock<HashSet<Pubkey>>,
// }

// impl<FG: ForkGraph> Default for TransactionBatchProcessor<FG> {
//     fn default() -> Self {
//         Self {
//             slot: Slot::default(),
//             epoch: Epoch::default(),
//             sysvar_cache: RwLock::<SysvarCache>::default(),
//             program_cache: Arc::new(RwLock::new(ProgramCache::new(
//                 Slot::default(),
//                 Epoch::default(),
//             ))),
//             builtin_program_ids: RwLock::new(HashSet::new()),
//         }
//     }
// }

// impl<FG: ForkGraph> TransactionBatchProcessor<FG> {
//     pub fn new_uninitialized(slot: Slot, epoch: Epoch) -> Self {
//         Self {
//             slot,
//             epoch,
//             program_cache: Arc::new(RwLock::new(ProgramCache::new(slot, epoch))),
//             ..Self::default()
//         }
//     }
//     pub fn just_wait(&self) {
//         sleep(Duration::from_millis(1000));
//     }
// }
















// use solana_program_runtime::loaded_programs::ForkGraph;

// use  crate::transaction_processor::{
//                 TransactionBatchProcessor,
//                 TransactionProcessingEnvironment,
//                 TransactionProcessingConfig,
//                 LoadAndExecuteSanitizedTransactionsOutput,
//             };


// pub fn new<FG: ForkGraph>(
//     slot: Slot,
//     epoch: Epoch,
//     fork_graph: Weak<RwLock<FG>>,
//     program_runtime_environment_v1: Option<ProgramRuntimeEnvironment>,
//     program_runtime_environment_v2: Option<ProgramRuntimeEnvironment>,
// ) -> TransactionBatchProcessor<FG: ForkGraph> {
//     let processor = TransactionBatchProcessor::new_uninitialized(slot, epoch);
//     {
//         let mut program_cache = processor.program_cache.write().unwrap();
//         program_cache.set_fork_graph(fork_graph);
//         processor.configure_program_runtime_environments_inner(
//             &mut program_cache,
//             program_runtime_environment_v1,
//             program_runtime_environment_v2,
//         );
//     }
//     processor
// }

// fn configure_program_runtime_environments_inner(
//     notSelf:&TransactionBatchProcessor<FG: ForkGraph>,
//     program_cache: &mut ProgramCache<FG: ForkGraph>,
//     program_runtime_environment_v1: Option<ProgramRuntimeEnvironment>,
//     program_runtime_environment_v2: Option<ProgramRuntimeEnvironment>,
// ) {
//     let empty_loader = || {
//         Arc::new(BuiltinProgram::new_loader(
//             VmConfig::default(),
//             FunctionRegistry::default(),
//         ))
//     };

//     program_cache.latest_root_slot = notSelf.slot;
//     program_cache.latest_root_epoch = notSelf.epoch;
//     program_cache.environments.program_runtime_v1 =
//         program_runtime_environment_v1.unwrap_or(empty_loader());
//     program_cache.environments.program_runtime_v2 =
//         program_runtime_environment_v2.unwrap_or(empty_loader());
// }

// use qualifier_attr::{field_qualifiers, qualifiers};
// use {
//     crate::{
//         account_loader::{
//             collect_rent_from_account, load_transaction, validate_fee_payer, AccountLoader,
//             AccountUsagePattern, CheckedTransactionDetails, LoadedTransaction,
//             TransactionCheckResult, TransactionLoadResult, ValidatedTransactionDetails,
//         },
//         account_overrides::AccountOverrides,
//         message_processor::MessageProcessor,
//         nonce_info::NonceInfo,
//         program_loader::{get_program_modification_slot, load_program_with_pubkey},
//         rollback_accounts::RollbackAccounts,
//         transaction_account_state_info::TransactionAccountStateInfo,
//         transaction_error_metrics::TransactionErrorMetrics,
//         transaction_execution_result::{ExecutedTransaction, TransactionExecutionDetails},
//         transaction_processing_callback::TransactionProcessingCallback,
//         transaction_processing_result::{ProcessedTransaction, TransactionProcessingResult},
//         transaction_processor::{
//             TransactionBatchProcessor,
//             TransactionProcessingEnvironment,
//             TransactionProcessingConfig,
//             LoadAndExecuteSanitizedTransactionsOutput,
//         },

//     },
//     log::debug,
//     percentage::Percentage,
//     solana_account::{state_traits::StateMut, AccountSharedData, ReadableAccount, PROGRAM_OWNERS},
//     solana_bpf_loader_program::syscalls::{
//         create_program_runtime_environment_v1, create_program_runtime_environment_v2,
//     },
//     solana_clock::{Epoch, Slot},
//     solana_compute_budget::compute_budget::ComputeBudget,
//     solana_compute_budget_instruction::instructions_processor::process_compute_budget_instructions,
//     solana_feature_set::{
//         enable_transaction_loading_failure_fees, remove_accounts_executable_flag_checks,
//         remove_rounding_in_fee_calculation, FeatureSet,
//     },
//     solana_fee_structure::{FeeBudgetLimits, FeeStructure},
//     solana_hash::Hash,
//     solana_instruction::TRANSACTION_LEVEL_STACK_HEIGHT,
//     solana_log_collector::LogCollector,
//     solana_measure::{measure::Measure, measure_us},
//     solana_message::compiled_instruction::CompiledInstruction,
//     solana_nonce::{
//         state::{DurableNonce, State as NonceState},
//         versions::Versions as NonceVersions,
//     },
//     solana_program_runtime::{
//         invoke_context::{EnvironmentConfig, InvokeContext},
//         loaded_programs::{
//             ForkGraph, ProgramCache, ProgramCacheEntry, ProgramCacheForTxBatch,
//             ProgramCacheMatchCriteria, ProgramRuntimeEnvironment,
//         },
//         solana_sbpf::{
//             program::{BuiltinProgram, FunctionRegistry},
//             vm::Config as VmConfig,
//         },
//         sysvar_cache::SysvarCache,
//     },
//     solana_pubkey::Pubkey,
//     solana_sdk::{
//         inner_instruction::{InnerInstruction, InnerInstructionsList},
//         rent_collector::RentCollector,
//     },
//     solana_sdk_ids::{native_loader, system_program},
//     solana_svm_rent_collector::svm_rent_collector::SVMRentCollector,
//     solana_svm_transaction::{svm_message::SVMMessage, svm_transaction::SVMTransaction},
//     solana_timings::{ExecuteTimingType, ExecuteTimings},
//     solana_transaction_context::{ExecutionRecord, TransactionContext},
//     solana_transaction_error::{TransactionError, TransactionResult},
//     solana_type_overrides::sync::{atomic::Ordering, Arc, RwLock, RwLockReadGuard},
//     std::{
//         collections::{hash_map::Entry, HashMap, HashSet},
//         fmt::{Debug, Formatter},
//         rc::Rc,
//         sync::Weak,
//     },
// };


// pub fn load_and_execute_sanitized_transactions<CB: TransactionProcessingCallback, FG: ForkGraph>( // literally main method, 
//     //<CB: TransactionProcessingCallback> 
//     // -> makes generic type CB over TransactionProcessingCallback
//     transaction_batch_processor: &TransactionBatchProcessor<FG>, // reference to transaction_batch_processor (TransactionBatchProcessor)
//     callbacks: &CB, // callbacks, type reference to CB, so bassicly this parameter should be an implementation of TransactionProcessingCallback !!!is very important to load accounts, bassicly each account should have owner, its data (amount of lamports, address, executable or not, etc) to be loaded -> for trnsactions to be proccessed 
//     sanitized_txs: &[impl SVMTransaction], // reference to an array of variables that implement the SVMTransaction trait, transactions that will be processed
//     check_results: Vec<TransactionCheckResult>, // vector of TransactionCheckResult, results of transaction checks
//     environment: &TransactionProcessingEnvironment, // runtime environment for transaction batch processing
//     config: &TransactionProcessingConfig, // config 
// ) -> LoadAndExecuteSanitizedTransactionsOutput {  // returns LoadAndExecuteSanitizedTransactionsOutput struct (error_metrics, execute_timings, processing_results)
// // If `check_results` does not have the same length as `sanitized_txs`,
// // transactions could be truncated as a result of `.iter().zip()` in
// // many of the below methods.
// // See <https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.zip>.
// debug_assert_eq!(    
// sanitized_txs.len(),      // macro than ensures that check_results.len() == sanitized_txs.len(), if they are not equal it will panic and print a message
// check_results.len(),      // already well described above
// "Length of check_results does not match length of sanitized_txs" // message that will be printed in case if check_results.len() == sanitized_txs.len() is not true
// );

// // Initialize metrics.
// let mut error_metrics = TransactionErrorMetrics::default(); // initializes default of this struct
// let mut execute_timings = ExecuteTimings::default();    // initializes default of this struct
// let mut processing_results = Vec::with_capacity(sanitized_txs.len()); // creates a vector with length (ccapacity) of sanitized_txs.len()

// let native_loader = native_loader::id(); // just an pubkey for native loader
// let (program_accounts_map, filter_executable_us) = measure_us!({ // custom macro measure_us measures execution time
//                           // and returns tuple of that what will be returned inside of macro and ex. time
//                           // and measured time in ms, in our case filter_executable_us - time in ms
//                           // and program_accounts_map - hash map, for executable accounts

// let mut program_accounts_map = transaction_batch_processor.filter_executable_program_accounts(  // filters all executable program accounts and returns hashmap with these accounts
// callbacks,
// sanitized_txs,
// &check_results,
// PROGRAM_OWNERS, // program that owns all executable programs
// );
// for builtin_program in transaction_batch_processor.builtin_program_ids.read().unwrap().iter() { // adds/pushes/inserts already built in programs to the program_accounts_map
// program_accounts_map.insert(*builtin_program, (&native_loader, 0));
// }
// program_accounts_map // returns program_accounts_map with executable accounts and the measured time will be stopped as this will be returned inside of macro
// });

// let (program_cache_for_tx_batch, program_cache_us) = measure_us!({ // custom macro measure_us measures execution time
//                      // and returns tuple of that what will be returned inside of macro and ex. time
// let program_cache_for_tx_batch = transaction_batch_processor.replenish_program_cache( // checks if some program accounts are missing in the cache
// // if yes, it loads them and returnes ProgramCacheForTxBatch, 
// // where inside are all programs needed for execution
// callbacks,
// &program_accounts_map,
// &mut execute_timings, // mutable reference
// config.check_program_modification_slot,
// config.limit_to_load_programs,
// );

// if program_cache_for_tx_batch.hit_max_limit {  // if cache is reached max limit of storage then error will be returned
// return LoadAndExecuteSanitizedTransactionsOutput {
// error_metrics,
// execute_timings,
// processing_results: (0..sanitized_txs.len()) // makes a range of length of sanitized_txs, iterates each with each index and for each retruns an error in processing_results
// .map(|_| Err(TransactionError::ProgramCacheHitMaxLimit)) // just small closure that takes whatever _ (index in this case) and returns TransactionError::ProgramCacheHitMaxLimit
// .collect(), // all iterators are "lazy" that is why we should always collect each iterator 
// };
// }

// program_cache_for_tx_batch // if cache isn't reached max limit of storage then it will be returned
// });

// // Determine a capacity for the internal account cache. This
// // over-allocates but avoids ever reallocating, and spares us from
// // deduplicating the account keys lists.
// let account_keys_in_batch = sanitized_txs.iter().map(|tx| tx.account_keys().len()).sum(); // pretty well described above :)
//                                             // rust concept: iterator over sanitized txs will be created
//                                             // then each transaction will be mapped and with the help of closure
//                                             //will be returned amount of all account keys in particular transaction
//                                             // at the end it will be summed and will be returned

// // Create the account loader, which wraps all external account fetching.
// let mut account_loader = AccountLoader::new_with_account_cache_capacity( // creates account loader with the capacity that we already calculated before
//                      // it "loads all accounts that are needed for execution of transactions"
// config.account_overrides,
// program_cache_for_tx_batch,
// program_accounts_map,
// callbacks,
// environment.feature_set.clone(),
// account_keys_in_batch,
// );

// let enable_transaction_loading_failure_fees = environment
// .feature_set                                // feature set, so whether there are some features turned on or not
// .is_active(&enable_transaction_loading_failure_fees::id()); // if these features are active than enable_transaction_loading_failure_fees will be true

// let (mut validate_fees_us, mut load_us, mut execution_us): (u64, u64, u64) = (0, 0, 0); // initializes default timings for validation, loading and execution

// // Validate, execute, and collect results from each transaction in order.
// // With SIMD83, transactions must be executed in order, because transactions
// // in the same batch may modify the same accounts. Transaction order is
// // preserved within entries written to the ledger.
// for (tx, check_result) in sanitized_txs.iter().zip(check_results) { // just a for loop that iterates over sanitized_txs (every single tx) 
//                                      // and check_results (corresponding to the tx result of checks)
// let (validate_result, single_validate_fees_us) = // measure_us -> validate_result and single_validate_fees_us
// measure_us!(check_result.and_then(|tx_details| {     // if err -> return err, if ok -> continue with closure
// transaction_batch_processor::validate_transaction_nonce_and_fee_payer( // ensures that transaction is nor repeating and that fee payer is provided and has enough funds
// &mut account_loader, // mutable reference
// tx,
// tx_details,
// &environment.blockhash,
// environment.fee_lamports_per_signature,
// environment
// .rent_collector
// .unwrap_or(&RentCollector::default()), // if rent_collector is not provided then use default
// &mut error_metrics, // mutable reference
// )
// }));
// validate_fees_us = validate_fees_us.saturating_add(single_validate_fees_us); // it will add time that it took for one transaction to validate fees to the sum of time 
//       // that it took for all transactions (0 by default)


// // load_transaction actually uses another function (bassicly is just a wrapper) called load_transaction_accounts which 
// // loads all accounts that are needed for execution of transactions, makes some additional checks and returns Result<LoadedTransactionAccounts> 
// // and the wrapper function load_transaction handles errors and returns an enum TransactionLoadResult
// let (load_result, single_load_us) = measure_us!(load_transaction(  // measure_us -> load_result and single_load_us
// &mut account_loader,  // mutable reference
// tx,
// validate_result,
// &mut error_metrics,  // mutable reference
// environment
// .rent_collector
// .unwrap_or(&RentCollector::default()), // if rent_collector is not provided then use default
// ));
// load_us = load_us.saturating_add(single_load_us);  // calculates time that was used for loading, simillar to validate_fees_us

// // exactly the execution of the transaction / processing 
// let (processing_result, single_execution_us) = measure_us!(match load_result { // measure_us -> processing_result and single_execution_us
// // it matches on the enum TransactionLoadResult
// TransactionLoadResult::NotLoaded(err) => Err(err), // if there was an error than return same error
// // FeesOnly is kind of tricky, as fas as I understood it is the case if the transaction fails during loading
// // and it is already too far and the fees should be charged though it failed
// TransactionLoadResult::FeesOnly(fees_only_tx) => {  
// if enable_transaction_loading_failure_fees {  // if the feature is enabled than it will be true
// // Update loaded accounts cache with nonce and fee-payer
// account_loader
// .update_accounts_for_failed_tx(tx, &fees_only_tx.rollback_accounts);

// Ok(ProcessedTransaction::FeesOnly(Box::new(fees_only_tx)))
// } else {
// Err(fees_only_tx.load_error) // if the feature is not enabled than return an error
// }
// }
// TransactionLoadResult::Loaded(loaded_transaction) => {  // if programs, accounts and transaction were loaded successfully
// // the transaction will be executed, all the instrcutions will be executed and all the balances will be changed
// let executed_tx = transaction_batch_processor.execute_loaded_transaction(
// callbacks,
// tx,
// loaded_transaction,
// &mut execute_timings, // mutable reference
// &mut error_metrics, // mutable reference
// &mut account_loader.program_cache, // mutable reference
// environment,
// config,
// );

// // Update loaded accounts cache with account states which might have changed.
// // Also update local program cache with modifications made by the transaction,
// // if it executed successfully.
// account_loader.update_accounts_for_executed_tx(tx, &executed_tx);

// Ok(ProcessedTransaction::Executed(Box::new(executed_tx)))
// }
// });
// execution_us = execution_us.saturating_add(single_execution_us); // measure time that was used for execution

// processing_results.push(processing_result); // push the result to the vector of results
// }

// // Skip eviction when there's no chance this particular tx batch has increased the size of
// // ProgramCache entries. Note that loaded_missing is deliberately defined, so that there's
// // still at least one other batch, which will evict the program cache, even after the
// // occurrences of cooperative loading.
// if account_loader.program_cache.loaded_missing
// || account_loader.program_cache.merged_modified
// {
// const SHRINK_LOADED_PROGRAMS_TO_PERCENTAGE: u8 = 90;
// transaction_batch_processor.program_cache
// .write()
// .unwrap()
// .evict_using_2s_random_selection(
// Percentage::from(SHRINK_LOADED_PROGRAMS_TO_PERCENTAGE),
// transaction_batch_processor.slot,
// );
// }

// // detailed logs on timings and amoUnt of txs
// debug!(
// "load: {}us execute: {}us txs_len={}",
// load_us,
// execution_us,
// sanitized_txs.len(),
// );


// // writing all timings
// execute_timings
// .saturating_add_in_place(ExecuteTimingType::ValidateFeesUs, validate_fees_us);
// execute_timings
// .saturating_add_in_place(ExecuteTimingType::FilterExecutableUs, filter_executable_us);
// execute_timings
// .saturating_add_in_place(ExecuteTimingType::ProgramCacheUs, program_cache_us);
// execute_timings.saturating_add_in_place(ExecuteTimingType::LoadUs, load_us);
// execute_timings.saturating_add_in_place(ExecuteTimingType::ExecuteUs, execution_us);

// // returning the results
// LoadAndExecuteSanitizedTransactionsOutput {
// error_metrics,
// execute_timings,
// processing_results,
// }
// }