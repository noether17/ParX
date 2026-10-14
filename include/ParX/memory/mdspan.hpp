#pragma once

#include <array>
#include <mdspan>
#include <type_traits>
#include <utility>

namespace ParX {

// CUDA currently requires that mdspan is indexed using operator() instead of
// operator[]. This class template provides the required interface.
template <typename T, typename Extents,
          typename LayoutPolicy = std::layout_right,
          typename AccessorPolicy = std::default_accessor<T>>
class mdspan {
 public:
  // Member types.
  using extents_type = Extents;
  using layout_type = LayoutPolicy;
  using accessor_type = AccessorPolicy;
  using mapping_type = LayoutPolicy::template mapping<Extents>;
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using index_type = Extents::index_type;
  using size_type = Extents::size_type;
  using rank_type = Extents::rank_type;
  using data_handle_type = AccessorPolicy::data_handle_type;
  using reference = AccessorPolicy::reference;

  // Static functions.
  static constexpr rank_type rank() noexcept { return Extents::rank(); }
  static constexpr rank_type rank_dynamic() noexcept {
    return Extents::rank_dynamic();
  }
  static constexpr std::size_t static_extent(rank_type r) noexcept {
    return Extents::static_extent(r);
  }
  static constexpr bool is_always_unique() {
    return mapping_type::is_always_unique();
  }
  static constexpr bool is_always_exhaustive() {
    return mapping_type::is_always_exhaustive();
  }
  static constexpr bool is_always_strided() {
    return mapping_type::is_always_strided();
  }

  // Constructors, destructor, and assignment operators.
  constexpr mdspan() = default;

  template <typename... OtherIndexTypes>
  constexpr explicit mdspan(data_handle_type p, OtherIndexTypes... exts)
    requires((std::is_convertible_v<OtherIndexTypes, index_type> and ...) and
             (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> and
              ...) and
             (sizeof...(OtherIndexTypes) == rank() or
              sizeof...(OtherIndexTypes) == rank_dynamic()) and
             (std::is_constructible_v<mapping_type, extents_type>) and
             (std::is_default_constructible_v<accessor_type>))
      : ptr_{std::move(p)},
        map_{extents_type{static_cast<index_type>(std::move(exts))...}} {}

  template <typename OtherIndexType, std::size_t N>
  constexpr explicit(N != rank_dynamic())
      mdspan(data_handle_type p, std::span<OtherIndexType, N> exts)
    requires((std::is_convertible_v<OtherIndexType const&, index_type>) and
             (std::is_nothrow_constructible_v<index_type,
                                              OtherIndexType const&>) and
             (N == rank() or N == rank_dynamic()) and
             (std::is_constructible_v<mapping_type, extents_type>) and
             (std::is_default_constructible_v<accessor_type>))
      : ptr_{std::move(p)}, map_{extents_type{exts}} {}

  template <typename OtherIndexType, std::size_t N>
  constexpr explicit(N != rank_dynamic())
      mdspan(data_handle_type p, std::array<OtherIndexType, N> const& exts)
    requires((std::is_convertible_v<OtherIndexType const&, index_type>) and
             (std::is_nothrow_constructible_v<index_type,
                                              OtherIndexType const&>) and
             (N == rank() or N == rank_dynamic()) and
             (std::is_constructible_v<mapping_type, extents_type>) and
             (std::is_default_constructible_v<accessor_type>))
      : ptr_{std::move(p)}, map_{extents_type{exts}} {}

  constexpr mdspan(data_handle_type p, extents_type const& ext)
    requires((std::is_constructible_v<mapping_type, extents_type const&>) and
             (std::is_default_constructible_v<accessor_type>))
      : ptr_{std::move(p)}, map_{ext} {}

  constexpr mdspan(data_handle_type p, mapping_type const& m)
    requires std::is_default_constructible_v<accessor_type>
      : ptr_{std::move(p)}, map_{m} {}

  constexpr mdspan(data_handle_type p, mapping_type const& m,
                   accessor_type const& a)
      : ptr_{std::move(p)}, map_{m}, acc_{a} {}

  template <typename OtherElementType, typename OtherExtents,
            typename OtherLayoutPolicy, typename OtherAccessor>
  constexpr explicit(
      not std::is_convertible_v<
          typename OtherLayoutPolicy::template mapping<OtherExtents> const&,
          mapping_type> or
      not std::is_convertible_v<OtherAccessor const&, accessor_type>)
      mdspan(mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy,
                    OtherAccessor> const& other)
    requires((std::is_constructible_v<
                 mapping_type, typename OtherLayoutPolicy::template mapping<
                                   OtherExtents> const&>) and
             (std::is_default_constructible_v<accessor_type>))
      : ptr_{other.ptr_}, map_{other.map_}, acc_{other.acc_} {}

  constexpr mdspan(mdspan const& rhs) = default;
  constexpr mdspan(mdspan&& rhs) = default;
  constexpr auto& operator=(mdspan const& rhs) = default;
  constexpr auto& operator=(mdspan&& rhs) = default;
  constexpr ~mdspan() = default;

  // Element access.
  template <typename... OtherIndexTypes>
  constexpr reference operator()(OtherIndexTypes... indices) const
    requires((std::is_convertible_v<OtherIndexTypes, index_type> and ...) and
             (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> and
              ...) and
             (sizeof...(OtherIndexTypes) == rank()))
  {
    return acc_.access(ptr_,
                       map_(static_cast<index_type>(std::move(indices))...));
  }
  template <typename OtherIndexType>
  constexpr reference operator()(
      std::span<OtherIndexType, rank()> indices) const
    requires(
        (std::is_convertible_v<OtherIndexType const&, index_type>) and
        (std::is_nothrow_constructible_v<index_type, OtherIndexType const&>))
  {
    auto access_at_span_indices =
        [&]<std::size_t... P>(auto span_indices, std::index_sequence<P...>) {
          return acc_.access(
              ptr_,
              map_(static_cast<index_type>(std::as_const(span_indices[P]))...));
        };
    return access_at_span_indices(indices, std::make_index_sequence<rank()>{});
  }
  template <typename OtherIndexType>
  constexpr reference operator()(
      std::array<OtherIndexType, rank()> const& indices) const
    requires(
        (std::is_convertible_v<OtherIndexType const&, index_type>) and
        (std::is_nothrow_constructible_v<index_type, OtherIndexType const&>))
  {
    return (*this)(std::span{indices});
  }

  // Observers.
  constexpr index_type extent(rank_type r) const noexcept {
    return extents().extent(r);
  }
  constexpr size_type size() const noexcept {
    auto size = size_type{1};
    for (auto r = rank_type{}; r < rank(); ++r) {
      size *= static_cast<size_type>(extent(r));
    }
    return size;
  }
  constexpr bool empty() const noexcept { return size() == 0; }
  constexpr index_type stride(rank_type r) const { return map_.stride(r); }
  constexpr extents_type const& extents() const noexcept {
    return map_.extents();
  }
  constexpr data_handle_type const& data_handle() const noexcept {
    return ptr_;
  }
  constexpr mapping_type const& mapping() const noexcept { return map_; }
  constexpr accessor_type const& accessor() const noexcept { return acc_; }
  constexpr bool is_unique() const { return map_.is_unique(); }
  constexpr bool is_exhaustive() const { return map_.is_exhaustive(); }
  constexpr bool is_strided() const { return map_.is_strided(); }

  friend constexpr void swap(mdspan& x, mdspan& y) noexcept {
    std::swap(x.ptr_, y.ptr_);
    std::swap(x.map_, y.map_);
    std::swap(x.acc_, y.acc_);
  }

 private:
  data_handle_type ptr_{};
  [[no_unique_address]] mapping_type map_{};
  [[no_unique_address]] accessor_type acc_{};
};

// Deduction guides.
template <typename CArray>
  requires(std::is_array_v<CArray> and std::rank_v<CArray> == 1)
mdspan(CArray&) -> mdspan<std::remove_all_extents_t<CArray>,
                          std::extents<std::size_t, std::extent_v<CArray, 0>>>;

template <typename Pointer>
  requires(std::is_pointer_v<std::remove_reference_t<Pointer>>)
mdspan(Pointer&&)
    -> mdspan<std::remove_pointer_t<std::remove_reference_t<Pointer>>,
              std::extents<std::size_t>>;

template <typename ElementType, typename... Integrals>
  requires((std::is_convertible_v<Integrals, std::size_t> and ...) and
           sizeof...(Integrals) > 0)
explicit mdspan(ElementType*, Integrals...)
    -> mdspan<ElementType, std::dextents<std::size_t, sizeof...(Integrals)>>;

template <typename ElementType, typename OtherIndexType, std::size_t N>
mdspan(ElementType*, std::span<OtherIndexType, N>)
    -> mdspan<ElementType, std::dextents<std::size_t, N>>;

template <typename ElementType, typename OtherIndexType, std::size_t N>
mdspan(ElementType*, std::array<OtherIndexType, N> const&)
    -> mdspan<ElementType, std::dextents<std::size_t, N>>;

template <typename ElementType, typename IndexType, std::size_t... ExtentsPack>
mdspan(ElementType*, std::extents<IndexType, ExtentsPack...> const&)
    -> mdspan<ElementType, std::extents<IndexType, ExtentsPack...>>;

template <typename ElementType, typename MappingType>
mdspan(ElementType*, MappingType const&)
    -> mdspan<ElementType, typename MappingType::extents_type,
              typename MappingType::layout_type>;

template <typename MappingType, typename AccessorType>
mdspan(typename AccessorType::data_handle_type const&, MappingType const&,
       AccessorType const&)
    -> mdspan<typename AccessorType::element_type,
              typename MappingType::extents_type,
              typename MappingType::layout_type, AccessorType>;

}  // namespace ParX
