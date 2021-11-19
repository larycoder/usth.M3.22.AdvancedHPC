#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise3/Exercise3.h>
#include <random>
#include <assert.h>

// ==========================================================================================
void Exercise3::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " [-m=m] [-n=n] [-o=o]"<< std::endl
        << "\twhere m is the row number of the first matrix," << std::endl
        << "\tand n is the column number of the first matrix, and row number of second matrix," << std::endl
        << "\tand o is the column number of the second matrix." << std::endl
        << std::endl;
}

// ==========================================================================================
void Exercise3::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise3::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise3& Exercise3::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    if( checkCmdLineFlag(argc, argv, "m") ) {
        int value = getCmdLineArgumentInt(argc, argv, "m");
        if( value > 1 )
            m = value;
    }
    if( checkCmdLineFlag(argc, argv, "n") ) {
        int value = getCmdLineArgumentInt(argc, argv, "n");
        if( value > 1 ) 
            n = value;
    }
    if( checkCmdLineFlag(argc, argv, "o") ) {
        int value = getCmdLineArgumentInt(argc, argv, "o");
        if( value > 1 ) 
            o = value;
    }
    return *this;
}


void Exercise3::run(const bool verbose) {    
    this->verbose = verbose;
    if( verbose )
        std::cout << std::endl << "Running exercise 3: build data" << std::endl;
    buildData();

    auto work = static_cast<StudentWork3*>(student);
    if( work->isQuestion1Implemented() )
        runQuestion1();
    if( work->isQuestion2Implemented() )
        runQuestion2();
    if( work->isQuestion3Implemented() )
        runQuestion3();
}

void Exercise3::runQuestion1() {     
    if( verbose )
        std::cout << std::endl << "***** Question: build a matrix" << std::endl;
    
    execute_and_display_GPU_time(verbose, [&]() -> void {
        auto& work = *static_cast<StudentWork3*>(student);
        if( sA != nullptr )
            delete sA;
        sA = new SparseMatrix(work.doQuestion1(m, n, matrixA));
    }, 3);
}

void Exercise3::runQuestion2() {
    if( verbose )
        std::cout << std::endl << "***** Question: sparse matrix times vector product" << std::endl;
    
    assert(sA != nullptr);

    execute_and_display_GPU_time(verbose, [&]() -> void 
    {
        auto& work = *static_cast<StudentWork3*>(student);
        OPP::CUDA::DeviceBuffer<double> dVector(aVector.data(), sA->getWidth());
        auto vector = work.doQuestion2(*sA, dVector);
        studentVector.resize(sA->getHeight());
        vector.copyToHost(studentVector.data());
    }, 3);
}

void Exercise3::runQuestion3() {
    if( verbose )
        std::cout << std::endl << "***** Question: product between two sparse matrices" << std::endl;
    
    assert(sA != nullptr);

    execute_and_display_GPU_time(verbose, [&]() -> void {
        auto& work = *static_cast<StudentWork3*>(student);
        if( sC != nullptr )
            delete sC;
        if( sB == nullptr )
            sB = new SparseMatrix(work.doQuestion1(n, o, matrixB));
        auto result = work.doQuestion3(*sA, *sB);
        sC = new SparseMatrix(result);
    }, 3);
}

bool Exercise3::check() {
    if( verbose )
        std::cout << std::endl << "====== do some checks ..." << std::endl;
    bool q1 = checkQuestion1();
    if( !q1 ) 
        std::cout << "--- question 1 not working " << std::endl;
    else
        std::cout << "--- question 1 works " << std::endl;
    bool q2 = checkQuestion2();
    if( !q2 ) 
        std::cout << "--- question 2 not working " << std::endl;
    else
        std::cout << "--- question 2 works " << std::endl;
    bool q3 = checkQuestion3();
    if( !q3 ) 
        std::cout << "--- question 3 not working " << std::endl;
    else
        std::cout << "--- question 3 works " << std::endl;
    return q1 && q2 && q3;
}

namespace {
    template<typename T, typename U>
    void display(T const*const array, const U size, const std::string& msg=std::string("display")) {
        if( size > 32 ) return;
        std::cout << msg << ": ";
        for(auto i=U(0); i<size; ++i)
            std::cout << array[i] << (i!=size-1?", ":"");
        std::cout << std::endl;
    }
}

bool Exercise3::checkQuestion1() {
    auto work = static_cast<StudentWork3*>(student);
    if( !work->isQuestion1Implemented() )
        return false;
    std::cout << "=== check question 1" << std::endl;
    // normally this should be ok ...
    if( sA->getHeight() != m ) return false;
    if( sA->getWidth() != n ) return false;
    const auto nbValues = matrixA.size();
    if( sA->values.getNbElements() != nbValues ) return false;
    if( sA->rows.getNbElements() != nbValues ) return false;
    if( sA->columns.getNbElements() != nbValues ) return false;
    if( sA->lookup.getNbElements() != m+1 ) return false;
    // now the student real work!
    {
        double *values = new double[nbValues];
        sA->values.copyToHost(values);
        //display(values, nbValues, std::string("values"));
        for(auto i=nbValues; i--;) if( values[i] != matrixA[i].value ) return false;
        delete values;
    }
    unsigned *rows = new unsigned[nbValues];
    sA->rows.copyToHost(rows);
    //display(rows, nbValues, std::string("rows"));
    for(auto i=nbValues; i--;) if( rows[i] != matrixA[i].row ) return false;
    {
        unsigned *columns = new unsigned[nbValues];
        sA->columns.copyToHost(columns);
        //display(columns, nbValues, std::string("columns"));
        for(auto i=nbValues; i--;) if( columns[i] != matrixA[i].column ) return false;
        delete columns;
    }
    {
        unsigned *lookup = new unsigned[m+1];
        sA->lookup.copyToHost(lookup);
        //display(lookup, m+1, std::string("lookup"));
        unsigned *goodLookup = new unsigned[m+1];
        for(auto i=0u; i<=m; ++i) goodLookup[i] = 0u;
        for(auto i=nbValues; i--;) goodLookup[ rows[i]+1 ] ++;
        for(auto i=1u; i<=m; ++i) goodLookup[i] += goodLookup[i-1];    
        for(auto i=m+1; i--;) if( lookup[i] != goodLookup[i] ) return false;
        delete lookup;
        delete goodLookup;
    }
    delete rows;
    return true;
}

bool Exercise3::checkQuestion2() {
    auto work = static_cast<StudentWork3*>(student);
    if( !work->isQuestion2Implemented() )
        return false;
    if( verbose )
        std::cout << "=== check question 2" << std::endl;
    if( studentVector.size() != m )
        return false;
    //display(aVector.data(), n, std::string("aVector"));
    //display(studentVector.data(), m, std::string("studentVector"));
    unsigned idx = 0u;
    for(auto row=0u; row<m; ++row) {
        double sum = 0.0;
        while( matrixA[idx].row == row ) {
            const auto& sv = matrixA[idx];
            sum += aVector[sv.column] * sv.value;
            idx ++;
        }
        if( fabs(sum - studentVector[row]) >= 1e-3 ) 
        {
            std::cout << "at "<<row<<" expect "<<sum<<" and got "<<studentVector[row]<<std::endl;
            return false;
        }
    }
    return true;
}

bool Exercise3::checkQuestion3() {
    auto work = static_cast<StudentWork3*>(student);
    if( !work->isQuestion3Implemented() )
        return false;
    std::cout << "=== question 3: check not implemented!" << std::endl;
    return true;
}


void Exercise3::buildData() 
{
    matrixA = generateASparseMatrix(m, n);
    matrixB = generateASparseMatrix(n, o);
    generateAVector(n, aVector);
}

std::vector<SparseValue> Exercise3::generateASparseMatrix(const unsigned numberOfRows, const unsigned numberOfColumns) 
{
    if( verbose ) 
        std::cout << "++++ build a sparse matrix of size (" << numberOfRows << "," << numberOfColumns << ")" << std::endl;

    std::uniform_real_distribution<> distributionValues(1.0, 10.0);
    std::uniform_int_distribution<> distributionRow(0, numberOfRows-1);
    std::uniform_int_distribution<> distributionColumn(0, numberOfColumns-1);

    std::vector<SparseValue> result;
    const auto nbDataPerRow = std::min(1u+(numberOfColumns/4u), 32u);
    const auto nbValues = 1u + nbDataPerRow * numberOfRows;

    for(auto i=nbValues; i--; ) {
        const SparseValue value(
            distributionValues(generator),
            distributionRow(generator),
            distributionColumn(generator)
        );
        bool exists = false;
        for(auto& sv : result) {
            if( sv.row == value.row && sv.column == value.column )
            {
                exists = true;
                break;
            }
        }
        if( ! exists )
            result.push_back(value);
    }

    std::sort(result.begin(), result.end(), [](const SparseValue&a, const SparseValue&b) {
        return (a.row < b.row) || ((a.row == b.row) && (a.column <= b.column));
    });

    return result;
}

void Exercise3::generateAVector(const unsigned size, std::vector<double>& vector) 
{
    if( verbose )
        std::cout << "++++ build a vector of size (" << size << ")" << std::endl;
        
    vector.clear();
    std::uniform_real_distribution<> distributionValues(0.0, 10.0);
    
    for(auto i=size; i--; ) 
        vector.push_back(distributionValues(generator));
    
    assert(vector.size() == size);
}