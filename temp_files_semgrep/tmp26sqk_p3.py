//! Utility macros for constructing an AST from an expression.

/// Constructs an `Expr` from a Nix expression.
///
/// # Limitations
///
/// Due to limitations of `macro_rules` in Rust, certain complex expressions might need extra
/// parentheses to reduce ambiguity and aid parsing. Some expressions are impossible to parse
/// without tweaking due to Rust's tokenization rules.
///
/// Also, if the compiler complains about hitting a certain recursion limit, try adding the
/// following module attribute to the root file of your crate:
///
/// ```rust,ignore
/// #![recursion_limit = "128"]
/// ```
///
/// # Example
///
/// ```rust,edition2018
/// # #![recursion_limit = "128"]
/// # use nix_parser::nix;
/// let expr = nix!({
///     inherit foo;
///     root = ./foo/bar;
///     config.value = {
///         first = true;
///         second = [1 2 -3];
///     };
/// });
/// ```
#[macro_export]
macro_rules! nix {
    ($($expr:tt)+) => {{
        #[allow(unused_imports)]
        use $crate::ast::*;
        #[allow(unused_imports)]
        use $crate::ast::tokens::*;
        $crate::nix_expr!($($expr)+)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! nix_token {
    (<$($template:tt)+>) => {
        Literal::Template(["<", $($template),+, ">"].into_iter().collect(), Default::default())
    };

    (/ $($path:tt)/+) => {
        Literal::Path(["/", $(stringify!($path)),+].into_iter().collect(), Default::default())
    };

    ($prefix:tt / $($path:tt)/+) => {
        Literal::Path([stringify!($prefix), $(stringify!($path)),+].into_iter().collect(), Default::default())
    };

    (false) => {
        Literal::Boolean(false, Default::default())
    };

    (true) => {
        Literal::Boolean(true, Default::default())
    };

    (null) => {
        Literal::Null(Default::default())
    };

    (_) => {
        Ident::from("_")
    };

    ($ident:ident) => {
        Ident::from(stringify!($ident))
    };

    ($($path:ident).+) => {
        AttrPath::new(vec![$(AttrSegment::Ident(Ident::from(stringify!($path)))),+])
    };

    ($literal:expr) => {
        Literal::from($literal)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! nix_list {
    (@elems (/ $($path:ident)/+) $next:tt $($rest:tt)*) => {{
        let first = $crate::nix_list!(@elems (/ $($path)/+));
        let rest = $crate::nix_list!(@elems ($next) $($rest)*);
        first.chain(rest)
    }};

    (@elems ($prefix:tt / $($path:tt)/+), $next:tt $($rest:tt)*) => {{
        let first = $crate::nix_list!(@elems ($prefix / $($path)/+));
        let rest = $crate::nix_list!(@elems ($next) $($rest)*);
        first.chain(rest)
    }};

    (@elems (- $expr:tt) $next:tt $($rest:tt)*) => {{
        let first = $crate::nix_list!(@elems (- $expr));
        let rest = $crate::nix_list!(@elems ($next) $($rest)*);
        first.chain(rest)
    }};

    (@elems (! $expr:tt) $next:tt $($rest:tt)*) => {{
        let first = $crate::nix_list!(@elems (! $expr));
        let rest = $crate::nix_list!(@elems ($next) $($rest)*);
        first.chain(rest)
    }};

    (@elems ($($prev:tt)*) / $next:tt $($rest:tt)*) => {
        $crate::nix_list!(@elems ($($prev)* / $next) $($rest)*)
    };

    (@elems ($($prev:tt)*) - $next:tt $($rest:tt)*) => {
        $crate::nix_list!(@elems ($($prev)* - $next) $($rest)*)
    };

    (@elems ($($prev:tt)*) ! $next:tt $($rest:tt)*) => {
        $crate::nix_list!(@elems ($($prev)* ! $next) $($rest)*)
    };

    (@elems ($expr:tt) $next:tt $($rest:tt)*) => {{
        let first = $crate::nix_list!(@elems ($expr));
        let rest = $crate::nix_list!(@elems ($next) $($rest)*);
        first.chain(rest)
    }};

    (@elems ($($expr:tt)+)) => {
        ::std::iter::once($crate::unary!($($expr)+))
    };

    (@elems ($($prev:tt)*) $next:tt $($rest:tt)*) => {
        $crate::nix_list!(@elems ($($prev)* $next) $($rest)*)
    };

    ([ ]) => {{
        #[allow(unused_imports)]
        use $crate::ast::tokens::*;
        #[allow(unused_imports)]
        use $crate::ast::*;
        ExprList::new(Vec::new(), Default::default())
    }};

    ([$first:tt $($rest:tt)*]) => {{
        #[allow(unused_imports)]
        use $crate::ast::tokens::*;
        #[allow(unused_imports)]
        use $crate::ast::*;
        ExprList::new($crate::nix_list!(@elems ($first) $($rest)*).collect(), Default::default())
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! nix_bind {
    (inherit $($names:ident)+) => {{
        let names = vec![$($crate::nix_token!($names)),+];
        Bind::Inherit(BindInherit::new(names, Default::default()))
    }};

    (inherit ($($expr:tt)+) $($names:ident)+) => {{
        let expr = $crate::nix!($($expr)+);
        let names = vec![$($crate::nix_token!($names)),+];
        Bind::InheritExpr(BindInheritExpr::new(expr, names, Default::default()))
    }};

    ($($name:ident).+ = $($expr:tt)+) => {{
        let attr = AttrPath::new(vec![$(AttrSegment::Ident(Ident::from(stringify!($name)))),+]);
        let expr = $crate::nix!($($expr)+);
        Bind::Simple(BindSimple::new(None, attr, expr, Default::default()))
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! nix_set {
    ({ }) => {
        ExprSet::new(Vec::new(), Default::default())
    };

    ({ $($binds:tt)+ }) => {
        ExprSet::new($crate::nix_set!(@binds $($binds)*).collect(), Default::default())
    };

    (@binds ($($bind:tt)+) ; $($rest:tt)+) => {
        ::std::iter::once($crate::nix_bind!($($bind)+)).chain($crate::nix_set!(@binds $($rest)+))
    };

    (@binds ($($bind:tt)+) ;) => {
        ::std::iter::once($crate::nix_bind!($($bind)+))
    };

    (@binds ($($prev:tt)*) $next:tt $($rest:tt)*) => {
        $crate::nix_set!(@binds ($($prev)* $next) $($rest)*)
    };

    (@binds $first:tt $($rest:tt)*) => {{
        #[allow(unused_imports)]
        use $crate::ast::*;
        #[allow(unused_imports)]
        use $crate::ast::tokens::*;
        $crate::nix_set!(@binds ($first) $($rest)*)
    }};
}

/// Returns an AST constructed by `nix!()` and also the source expression as a static string.
///
/// This is useful for testing whether the parser and the `nix!()` macro both produce the same AST.
///
/// # Example
///
/// ```rust,edition2018
/// # use nix_parser::nix_expr_and_str;
/// # use nix_parser::ast::{tokens::Literal, Expr};
/// #
/// let (expr, s) = nix_expr_and_str!(~/foo/bar);
/// assert_eq!(expr, Expr::Literal(Literal::Path("~/foo/bar".into(), Default::default())));
/// assert_eq!(s, "~/foo/bar");
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! nix_expr_and_str {
    ($($expr:tt)+) => {
        (
            $crate::nix_expr!($($expr)+),
            stringify!($($expr)+)
                .replace("< ", "<")
                .replace(" >", ">")
                .replace(" . ", ".")
                .replace(" ;", ";")
                .replace("- ", "-")
                .replace("! ", "!")
                .replace("/ /", "//")
                .replace(" / ", "/")
                .replace("$ ", "$")
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! nix_expr {
    ($($expr:tt)+) => {
        $crate::unary!($($expr)+)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! unary {
    (@rule - $($expr:tt)+) => {
        Expr::Unary(Box::new(ExprUnary::new(UnaryOp::Neg, $crate::atomic!($($expr)+), Default::default())))
    };

    (@rule ! $($expr:tt)+) => {
        Expr::Unary(Box::new(ExprUnary::new(UnaryOp::Not, $crate::atomic!($($expr)+), Default::default())))
    };

    (@rule $($expr:tt)+) => {
        $crate::atomic!($($expr)+)
    };

    ($($expr:tt)+) => {{
        #[allow(unused_imports)]
        use $crate::ast::*;
        #[allow(unused_imports)]
        use $crate::ast::tokens::*;
        $crate::unary!(@rule $($expr)+)
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! atomic {
    (($($expr:tt)+)) => {
        Expr::Paren(ExprParen::new(Box::new($crate::nix_expr!($($expr)+)), Default::default()))
    };

    ({$($binds:tt)*}) => {
        Expr::Set($crate::nix_set!({$($binds)*}))
    };

    ([$($expr:tt)*]) => {
        Expr::List($crate::nix_list!([$($expr)*]))
    };

    (<$($template:tt)+>) => {
        Expr::Literal($crate::nix_token!(<$($template)+>))
    };

    (/ $($path:tt)/+) => {
        Expr::Literal($crate::nix_token!(/$($path)/+))
    };

    ($prefix:tt / $($path:tt)/+) => {
        Expr::Literal($crate::nix_token!($prefix/$($path)/+))
    };

    (false) => {
        Expr::Literal($crate::nix_token!(false))
    };

    (true) => {
        Expr::Literal($crate::nix_token!(true))
    };

    (null) => {
        Expr::Literal($crate::nix_token!(null))
    };

    (_) => {
        Expr::Ident($crate::nix_token!(_))
    };

    ($attr:ident) => {
        Expr::Attr($crate::nix_token!($attr))
    };

    ($literal:expr) => {
        Expr::Literal($crate::nix_token!($literal))
    };

    ($($err:tt)*) => {
        compile_error!(stringify!($($err)*));
    };
}
